# 02. Harness: training stack, reward fleet and verification modes

**Status (Oct 24, 2026): plan.** Tags: [V] verified, [R] repo, [L] unverified literature or general knowledge, [P] proposal.

## Summary
- **Today's bottleneck is the reward path, not the GPU.** Run 1 averaged 196 s per step (173 s → 199 s over the
  run) for 128 completions [R]. Rewards go to one macOS VM with 8 workers through an SSH tunnel from the Mac [R]. Every
  published code-RL harness puts a router with a global concurrency cap in front of many workers and scores each
  completion as it finishes [V].
- **The ideal stack has three disaggregated parts** [V]:
  1. vLLM generation with in-flight weight updates (2.17x throughput on its own in OlmoRL);
  2. a trainer fed asynchronously, with bounded staleness (η ≤ 4 for code in AReaL) and an off-policy-corrected objective;
  3. a scale-out reward fleet.

  Async gives 1.5–2.3x end-to-end against strong synchronous baselines at 1.5–32B, with matched accuracy [V].
  Overlap alone is capped at 2x [V].
- **Every target ISA needs its own execution backend.** Linux AArch64 is a different target from Apple arm64, not a
  drop-in replacement: x18 is reserved on Apple, every variadic argument goes on the stack, and `va_list` is `char*` [V].
  GG ran its ARMv8 tests natively on an M2 Pro. It never says how it ran RISC-V tests [V]. So GG comparisons keep a
  macOS reward target, and we choose the RISC-V method ourselves.
- **Hacking defenses and per-verdict telemetry are part of the harness.** False-positive rewards cause sudden
  collapses in code RL [V] ([04](04-synthetic-tests.md)).

## 1. What run 1 tells us [R]
`results/training/full/trainer_state.json`, 106 steps:

| Metric | First 10 steps | Last 10 steps |
|---|---|---|
| step time (s) | 174 | 199 |
| reward (pass rate) | 0.40 | 0.49 |
| `frac_reward_zero_std` (groups with no signal) | 0.58 | 0.56 |
| entropy | 0.038 | 0.028 |
| mean completion length (tokens) | 982 | 1,010 |
| truncated (clipped) | 0.6% | 2.3% |
| verdict: crash (includes failed asserts, which abort) | 0.42 | 0.35 |
| verdict: assemble_err | 0.15 | 0.13 |
| verdict: timeout | 0.036 | 0.030 |

The numbers point to four things:
1. **Signal is wasted.** 57% of groups give zero gradient, which is a curriculum problem ([03](03-curriculum.md)).
2. **Exploration is low.** Entropy of about 0.03 means many identical completions, so a verdict cache keyed on the
   normalized asm should hit often. Measure it first.
3. **The step-time split is unknown.** No log separates generation, scoring and training time. Instrument the harness
   before choosing between async overlap (at most 2x) and scaling out the rewards [V].
4. **A crash verdict mixes assert aborts with segfaults.** Partial-credit or ladder rewards need the split
   (experiments.md, R2).

## 2. Target architecture (ideal resources)
```
                 ┌──────────── weights (NCCL, in-flight) ────────────┐
                 ▼                                                   │
  prompts ─► vLLM generation pool ─► completions ─► reward router ─► trainer (FSDP2/ZeRO, fp32 master)
  (curriculum     (N GPUs)            (streamed)     (global cap,      (M GPUs; bounded-staleness
   sampler)                                          cache, retries)    buffer η ≤ 4)
                                                          │
                       ┌──────────────────┬───────────────┴───┬─────────────────────┐
                       ▼                  ▼                   ▼                     ▼
                 macOS arm64 hosts   Linux AArch64 hosts   RISC-V64 (QEMU-user    x86-64 hosts
                 (clang 14, SDK 13.3) (Graviton/Ampere)    or boards)             (reference runs,
                                                                                  synthetic-test gates)
```
- **Framework.** Use one framework for every confirmatory run. Changing it mid-study breaks the
  "same everything except one switch" rule ([05](05-statistics.md)).
  - Candidates with published async support are verl, AReaL, prime-rl, slime, SkyRL and OpenRLHF [L]. Of these, only the
    AReaL, PipelineRL, DeepCoder (verl-pipeline) and OlmoRL numbers are verified [V].
  - TRL 1.16 `AsyncGRPOTrainer` overlaps generation and scoring with training. It needs vLLM and the trainer on separate
    GPUs, FSDP2 only, NCCL weight transfer and transformers ≥ 5.2. It is experimental, and no speedup is published [V].
  - [P] Pick by a one-day bake-off:
    1. Run the same recipe for 50 steps on the current TRL sync stack and on the candidate in strict on-policy mode.
       The learning curves must agree within noise, which checks that both implement the same estimator.
    2. Turn on async (η = 1, 2, 4) and compare throughput and the curve.
    3. Keep the candidate only if its curve matches TRL's at η = 0.
- **Staleness and objective.**
  - With AReaL's decoupled objective, η ≤ 8 barely moves accuracy. Without it, η = 4 fell from 42.0 to 23.3 AIME24 in one
    ablation [V, single run].
  - Default η = 1 for the method study, so results stay close to on-policy. Raise it only for scale runs, and report η
    for every run [P].
  - Clipping variants (`epsilon_high`, CISPO, GSPO) only act off-policy. Our sync runs were exactly on-policy, with clip
    ratios of 0 at every step [R].
- **Logits in FP32 in both the trainer and the rollout engine.** In ScaleRL at 8B this raised the asymptote from 0.52 to
  0.61 with DAPO-style losses [V, split vote]. It is cheap; turn it on first.

## 3. Reward fleet
### 3.1 Router [V pattern, P details]
- One stateless router on a CPU node. It takes batches, deduplicates them, consults the cache, dispatches with a global
  concurrency cap and per-host caps, retries on infra errors, and returns verdicts as each one finishes.
  - open-r1 uses an asyncio semaphore (Morph default 20) [V].
  - A per-endpoint limit alone overloads workers (Piston) [V].
- **Cache key:** (task ID, target, toolchain hash, normalized asm hash). Normalization removes comments and blank lines
  and canonicalizes whitespace; labels stay. Cached verdicts are bit-identical by construction, because the toolchain
  hash is in the key.
- **Early exit for the binary reward:** stop at the first failing test. Partial-credit rewards run all tests.
- **Test capping in training** [R proposal]: contest problems have a median of 41–46 official tests. Training uses a
  fixed subset of 20 per problem, chosen once and stratified so the largest tests are kept. Eval and dev always run the
  full set.
- **Telemetry per verdict:** host, target, wall time for assemble, link and run, cache hit, and exit status. These give
  the step-time split from §1.

### 3.2 Execution backends
| Target | Toolchain | Where it runs | Scaling notes | Tag |
|---|---|---|---|---|
| `arm64-apple-macos13` (GG comparability) | Apple clang 14.0.3, SDK 13.3 (reproduces GG's assembly byte for byte) | Apple Silicon. Now: Tart VM `gg-ventura`. At scale: macOS hosts running workers on bare metal under `sandbox-exec`, or 2 VMs per host | The macOS EULA allows at most 2 macOS VMs per Mac, so scale by hosts or run on bare metal. Cloud Macs (EC2 Mac) exist with long minimum allocations | [R] toolchain; [L] EULA and cloud |
| `aarch64-linux-gnu` | clang (pin the version) + glibc | Graviton/Ampere natively in containers (nsjail/bubblewrap/seccomp), Linux VMs on Apple Silicon, or QEMU-user on x86 | native ARM Linux is the cheapest scale-out; ABI differs from Apple [V] | [V] ABI; [L] rest |
| `riscv64-linux-gnu` | clang or GCC cross (GG used `gcc-riscv64-linux-gnu`) | QEMU-user on x86/ARM Linux; real boards only for timing | GG gives no precedent for executing RISC-V [V] | [V][L] |
| `x86_64` references | clang at the same version | Linux x86 | needed for the synthetic arm's cross-ISA agreement gate ([04](04-synthetic-tests.md)) | [P] |

**Compiler choice for the redo [P]:**
- Use one compiler family (clang) for every target, at one pinned version per platform, so differences between targets
  are about the ISA, not the compiler.
- For RISC-V, GG used GCC. To compare with GG on RISC-V, also rebuild GG's RISC-V benchmark rows with GCC. Report both
  numbers and say why.

**ABI and assembler differences to handle [V for ABI, L for syntax]:**
- **Apple:**
  - x18 is reserved;
  - variadic arguments go in 8-byte stack slots;
  - `va_list` is `char*`;
  - symbols carry an underscore prefix;
  - Mach-O directives (`.section __TEXT,__text`) and `@PAGE`/`@PAGEOFF` relocations are used.
- **Linux:**
  - x18 is caller-saved;
  - variadic arguments go in x0–x7/v0–v7;
  - ELF directives and `:lo12:` relocations are used.
- **Proposal:** prompts name the target. A static checker rejects x18 use on Apple targets with a distinct verdict
  (`abi_violation`). Data are labeled by target.

### 3.3 Sizing (estimate; measure before buying) [P]
- ScaleBox measured about 0.6 Python tasks per second per core on one node, and scaling across nodes was sub-linear [V].
- Our cost per completion is roughly one assemble and link (on the order of 0.1 s) plus k test runs of a few ms each,
  plus sandbox start-up. All of this is unmeasured.
- At 512 completions per step with 20 tests each, that is roughly 100 CPU-seconds per step: a few seconds on 64 cores, and
  tens of seconds on the current 8 VM workers.
- **Conclusion:** at ideal scale, reward cost is small next to generation. At our scale, the VM and the tunnel are the
  limit. One more Apple Silicon host with bare-metal workers would likely remove it.

## 4. Verification modes
| Mode | Prompt unit | Oracle | Reward | Status |
|---|---|---|---|---|
| **driver** | one function (`func0`) | C driver with asserts, linked with the prediction | pass/fail | in use [R] |
| **stdio** | whole program (`main`) | official stdin → stdout, compared exactly except trailing whitespace | pass/fail; optionally the fraction of tests passed | to build [R design] |
| **splice** | one function inside a whole program | link the predicted function with the **compiler's** versions of every other function, then run the program's real tests | per-function pass, giving dense, real-test-based credit for whole programs | new [P] |
| **differential** | function | reference build on generated valid inputs, compared on return value, reachable memory, globals and stdout | pass/fail plus adequacy metadata | synthetic arm [P] ([04](04-synthetic-tests.md)) |
| **equivalence check** | function | symbolic: angr + z3 against the reference | eval-only signal | D-LiFT returns a result for 80.7% of functions, with precision 0.91 measured on 100 [V] |

**Splice mode is the harness feature that the curriculum and partial credit depend on [P].**
- It turns a whole program into many function-level tasks, all graded by the program's real tests. A wrong function
  fails the program's tests, given that the compiler's versions of the other functions are correct.
- Credit for a test is exact. If a function is never exercised by the tests, a wrong version of it can pass, so report
  coverage per function from the reference run.
- It needs symbol-level linking: assemble the predicted function and the compiler's object for the rest of the program,
  with no duplicate definitions. On Mach-O this needs care with local labels and literal pools.

## 5. Reward hacking and robustness
**Static checks before linking [P].** Each failure gets its own verdict, so it shows up in the logs:
- no `svc`/`syscall` and no inline system calls;
- no definitions of symbols the task does not ask for, such as `main` in driver mode or `func0` in stdio mode;
- no `.incbin` or `.include`;
- no x18 on Apple targets;
- no writable sections where the reference has none;
- a size limit (the completion cap already bounds it).

**Dynamic checks:**
- sandbox (`sandbox-exec` on macOS today [R]; nsjail/bubblewrap/seccomp on Linux [L]);
- limits on CPU seconds, memory and file size [R];
- no network and a fresh working directory;
- exact output compare;
- stdio tasks need at least 2 distinct expected outputs [R];
- run every passing completion twice, and count disagreement as an infra flag, not a pass.

**Audits:**
- Stub audit: score `mov w0,#0; ret`, `ret` and "print a constant" on every task. Any stub that passes means the task
  is dropped or its tests are fixed [R plan].
- Monitor the stub rate, verdict mix, length and dev pass rate on **full** tests against training pass rate on
  **capped** tests. A widening gap means the policy is overfitting the cap.

**Why this matters:**
- In AceReason-7B, mixing in weak-test problems caused a sudden mid-training drop in held-out accuracy [V].
- HardTests vs TACO tests on the same problems: +7.6 pp pass@10 [V, single runs].

## 6. Measuring code efficiency (for the efficiency objective)
- GG measured execution time, energy (powermetrics) and memory against Rosetta and native code, with the geometric mean of
  100 runs [R, paper §5.2].
- **Deterministic cost for rewards [P].**
  - Wall-clock is too noisy to use as a reward.
  - Use the dynamic instruction count and the cycle estimate from a QEMU TCG plugin or `perf stat -e instructions` on
    pinned cores, plus static `.text` size.
  - For each passing completion, define r_eff = clip(log(cost_ref / cost), −c, c) against the clang reference at the same
    optimization level.
- **Wall-clock and energy for reporting only:** dedicated hosts, pinned cores, at least 30 repetitions, geometric mean
  with a bootstrap CI.
- **Feasibility:**
  - ARM Linux has `perf` [L]. QEMU plugins cover AArch64 and RISC-V [L].
  - On macOS, energy needs `powermetrics`, as in GG, and instruction counts need a different tool. That is open
    question Q4 in §8.

## 7. Reproducibility
- Pin the containers or VM images, and record the toolchain hash in every verdict.
- Record the config, data hash, seed and git SHA for every run (W&B already logs the run config [R]).
- **Known hazard [R]:** checkpoints saved by transformers 5.x have no top-level `rope_theta`, and 4.x then silently uses
  10000. Assert it at load time, which `generate.py` and the cluster `gen_bringup.py` already do.
- Every base-vs-trained comparison uses one GPU type and one software stack [R].

## 8. Open questions
- Q1. What is the step-time split (generation, scoring, training)? Instrument before choosing a lever [V].
- Q2. What is the verdict-cache hit rate at our entropy?
- Q3. How fast and how faithful is QEMU-user for AArch64 and RISC-V on assemble+link+run? Answer it with a reference
  pass-rate check: every reference passes under QEMU exactly when it passes natively.
- Q4. What deterministic cost counter can we use on macOS for the efficiency arm?
- Q5. Does TRL `AsyncGRPOTrainer` correct off-policy samples, or only drop the stale ones? The claim about its staleness
  defaults was refuted [V].

## Sources
Verified claims: [evidence/02-harness.md](evidence/02-harness.md).
- AReaL: arXiv 2505.24298; PipelineRL: arXiv 2509.19128; Periodic Asynchrony: arXiv 2511.18871; DeepCoder:
  together.ai/blog/deepcoder; OlmoRL / Olmo 3: arXiv 2512.13961
- TRL AsyncGRPOTrainer: https://huggingface.co/docs/trl/async_grpo_trainer
- ScaleBox: arXiv 2604.27467; open-r1: https://github.com/huggingface/open-r1
- Apple arm64 ABI: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
- ScaleRL (FP32 logits): arXiv 2510.13786; D-LiFT: arXiv 2506.10125; AceReason: arXiv 2505.16400; HardTests: arXiv 2505.24098
