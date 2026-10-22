> Superseded by status-and-plan.md. This is the running planning log. Early sections predate later findings (for example the MBPP-C counts, the ponyo run, and the 164 vs 158 decision), so treat later "Update" sections as authoritative.

# Simple QLoRA-GRPO plan for Guaranteed-Guess (research and planning only)

## 1. Where the original project lives

- Local git branch `qlora-grpo-study` in `~/git/Guaranteed-Guess` on Enzo's Mac (HEAD `c5b4c60`, two commits on top of `main`).
- It is **not on GitHub**: `origin` only has `main`, and the cloud repo checkout has no `rl/` folder. No earlier chats or tasks exist in this project, so the code and its docs are the only record.
- Two commits: "Add pre-registered QLoRA-GRPO study for O2/O3 transpilation" and "Add Slurm support, CPU dry run, and fix chat-template token counts".
- Uncommitted edits: ExeBench sharded/resumable build (`rl/data/build_exebench.py`), BringUp support in `rl/reward/harness.py` and `rl/reward/server.py`.
- Size: 109 files, about 2.8k lines of Python/shell in `rl/`, plus `paper/` (LaTeX skeleton, results sections are still `\todo`).

## 2. What it contains

| Piece | What it is | Verdict |
|---|---|---|
| `rl/reward/{harness,server,client,extract}.py`, `sandbox.sb` | Assemble predicted ARM64 `.s`, link with a test driver, run sandboxed, return pass/fail. HTTP batch server on the Mac, TRL reward function on the GPU box. | **Keep as is.** This is the one part nobody else has written. |
| `rl/data/build_humaneval.py`, `build_exebench.py`, `build_bringup.py`, `rl/artifacts/data/*.parquet`, `data_cache/` | Test-bearing datasets: HumanEval-C (164), MBPP-C (928 after drops), ExeBench valid/test built; ExeBench train has 9 shards cached but is not merged. | **Keep.** Data is already built (59 MB parquet, 811 MB cache). |
| `rl/qrl/train.py` | TRL `GRPOConfig` + `GRPOTrainer` wiring, YAML merge, bnb config, LoRA config. | **Keep the idea, shrink to about 60 lines.** |
| `rl/qrl/trainer.py` (subclass), `sync.py` (custom vLLM weight sync), `quant.py` (fake NF3/INT8/noise), `prepare_bases.py`, `parity_check.py`, `token_classes.py` | Needed only for the full pre-registered study (separate sampler base, perturbation channels). | **Drop for now.** About 500 lines of custom code that wrap TRL internals (`_vg.is_bitsandbytes_available` monkeypatch, `_iter_named_params` override). |
| `rl/configs/arms/*.yaml` (20 arms), `PREREG.md`, `analysis/`, `slurm/`, `scripts/` | 3-tier, about 60-run design with TOST, Holm, mediation analysis. | **Park.** Reuse two arms only. |
| `rl/docker/Dockerfile.armv5` | ARMv5 gcc-10 + QEMU replication. | **Park** (Tier 3 only). |
| `train/compile_{x86,arm}_clang.py`, `eval/` (upstream GG) | Original data curation and eval scripts. | Reference only. |

## 3. Why the toolchain is being rebuilt (my reading of the code, please confirm)

- GG's ARMv8 data is Mach-O output of `clang -arch arm64`, so predicted assembly can only be assembled, linked and run natively on Apple Silicon. The training box is x86 Linux with RTX 6000s, so the branch builds a reward server on the Mac and a reverse SSH tunnel (`ssh -R 8765`) back to the GPU box or Slurm login node.
- The upstream eval (`eval/bringup-bench` Makefile) compiles the C source of the function into the test binary, so an empty prediction could pass silently. `harness.py` instead makes the predicted assembly the only definition of the function, with empty/truncated outputs failing. That is a correctness requirement for RL, because a hackable reward would be exploited.
- The rest of the rebuild (Tart VM, `.venv-mac`, ARMv5 Docker, `setup_gpu.sh`, Slurm wrappers) exists to run the full 60-run study. None of it is needed for a first experiment.

## 4. Library research (what is maintained and what it covers)

- **TRL `GRPOTrainer`** (current docs): `loss_type` options `grpo`, `dapo`, `dr_grpo`, `sapo`; asymmetric clip via `epsilon`/`epsilon_high`; `beta=0` default; vLLM `colocate` mode with `vllm_enable_sleep_mode`, or `server` mode on separate GPUs; truncated importance sampling is on by default for vLLM and logs `sampling/*` mismatch metrics; reward functions are plain Python callables `(prompts, completions, **dataset_columns) -> list[float]` and can be `async`. All of what the old `GGGRPOTrainer` logs for mismatch is already in TRL.
- **PEFT + bitsandbytes via TRL**: GRPOTrainer accepts `peft_config` and `quantization_config` directly. TRL docs suggest LoRA GRPO learning rate about 1e-5.
- **Not documented, so must be smoke-tested**: (a) how TRL pushes LoRA weights into vLLM when the base is 4-bit, (b) FSDP2 + bnb 4-bit under GRPO (TRL's GRPO docs show DeepSpeed ZeRO-3, not FSDP). These two gaps are what motivated the custom `sync.py` and the FSDP1/FSDP2 fallback configs. For a 1.5B-class model, QLoRA fits on one 24 GB card, so neither is needed in a first run.
- **Hardware caveat**: Turing sm_75 means fp16 only, no bf16, no FlashAttention-2 (the old branch already uses `sdpa`). vLLM support for sm_75 in the pinned range (>=0.22,<=0.32) should be verified in a 10-step smoke run.

## 5. Proposed simple plan

**Goal of v0:** one honest result, LoRA-GRPO vs QLoRA-GRPO on `gg-armv8-O2`, using only maintained libraries.

1. **Freeze what exists.** Push `qlora-grpo-study` to GitHub as-is (commit the uncommitted edits on top), so it is backed up and visible to cloud sessions. Nothing is deleted.
2. **New slim directory** (for example `rl_simple/`, about 150 lines in total):
   - `reward.py`: import `reward.harness`/`client.RewardFn` unchanged.
   - `train.py`: `GRPOConfig(loss_type="dapo", use_vllm=True, vllm_mode="colocate", ...)`, `LoraConfig`, optional `BitsAndBytesConfig`, `GRPOTrainer(model=..., peft_config=..., reward_funcs=RewardFn())`. No subclass, no custom sync.
   - `eval.py`: sample from the saved adapter with vLLM, score through the same reward server, report pass@1 on HumanEval-C O2 and a held-out slice.
3. **Hardware:** a single RTX 6000 first (no FSDP, no accelerate multi-process). Add the second GPU only if throughput demands it.
4. **Data:** MBPP-C + HumanEval-C pool with a fixed split, no difficulty-filter pipeline (just drop prompts where the base model solves 0/8 or 8/8 in a short pre-pass, or skip filtering entirely for v0).
5. **Arms:** A0 = fp16 LoRA, A1 = NF4 QLoRA, 1 seed each, 100-300 steps, same hyperparameters. Add seeds or DAPO vs GRPO only after A0/A1 train stably.
6. **Smoke tests (before any long run):** 10 steps with the real reward server over the tunnel; confirm vLLM sees updated weights (reward or logprobs change after step 1); confirm NF4 sync works. If NF4 sync fails, that is when to consider the old `sync.py`, not before.
7. **Logging:** TRL's built-in W&B/TensorBoard metrics only (reward, entropy, clip ratio, `sampling/*` mismatch).

Estimated custom code: reward reuse (0 new lines) + about 60-line `train.py` + about 50-line `eval.py`, versus about 2.8k lines in the old branch.

## 6. Risks to watch

- Mac availability throughout runs (reward throughput was estimated at 7-15 items/s in the old README, which caps steps/hour).
- vLLM on Turing and TRL-with-4-bit-base weight sync are undocumented combinations.
- Test leakage: ExeBench train overlaps `test_real` (old branch deduplicates by hash), and 6 HumanEval problems have inconsistent reference ARM.

---

# Update after Enzo's answers (baseline-first plan)

## Facts checked
- `ahmedheakl/gg-armv8-O2` is **Qwen2.5-Coder-1.5B-Instruct already SFT'd** (LLaMA-Factory, 2 epochs, lr 2e-5, effective batch 192 on 8 GPUs). Its SFT data is the ~1.31M-pair `ahmedheakl/gg-armv8-O2-instruct` (AnghaBench + Stack functions). So **no SFT is needed**: GRPO starts from this checkpoint, which already produces the right output format. Re-running SFT would cost 8-GPU-scale compute for nothing.
- The SFT data has no executable tests, so it cannot give a pass/fail reward. RL prompts need tests: HumanEval-C and MBPP-C (assert tests) and ExeBench (real I/O). Those are what the old branch already built. The SFT set is still useful as a source of extra x86 prompts only if tests are generated for them, which I do not recommend for v0.
- Midpoint report (earlier thread) says the project topic is now GRPO + QLoRA on Guaranteed-Guess. The VLA/DAPO proposal in `CS8001/` is the old topic.

## Baseline v0 (what to run first)
1. **Step-0 baseline eval, no training:** run `gg-armv8-O2` on HumanEval-C O2 through the existing reward harness, to confirm the harness reproduces the paper's ~45% (beam-8) and gives a sampled pass@1 at T=1. This is the "replicate the paper" part and costs only inference.
2. **One GRPO run, bf16 LoRA, TRL `GRPOTrainer`** with `loss_type="dapo"` (a single flag; it is the DAPO token-level loss with asymmetric clip, not the full DAPO recipe), G=8, binary reward from the Mac server, on MBPP-C (+ ExeBench valid/train slice), 100-300 steps. On the 96 GB Blackwell this is one GPU, no FSDP, no accelerate multi-process.
3. **Same run with NF4 QLoRA** (change `quantization_config` only). On 96 GB, QLoRA is not needed for memory, so it is purely the research variable.
4. **Eval both on HumanEval-C O2** with the same harness. W&B logs everything TRL emits.

## Blackwell notes
- RTX 6000 Blackwell is sm_120: needs a recent PyTorch (CUDA 12.8+) and a vLLM build with Blackwell support. bf16 and FlashAttention-class kernels work, so the Turing workarounds (fp16-only, sdpa) in the old branch go away. This must be verified with a 10-step smoke run.
- 1.5B at 96 GB leaves room for full fine-tuning too, but I recommend LoRA first because it is what the project is about.

## Where to run
- Training and vLLM on the lab GPU host; reward server stays on the Mac; one reverse tunnel (`ssh -R 8765:127.0.0.1:8765 <lab-host>`) so the trainer calls `127.0.0.1:8765`. If the lab host is a Slurm login node, a single `sbatch` with that tunnel is enough; the old `slurm/` wrappers are not needed for a one-GPU job (an interactive `srun` or a 20-line sbatch works).
- "Per-token-class probes" in the old branch meant measuring entropy separately for mnemonics, registers and immediates in the generated assembly. That was for the pre-registered analysis; drop it.

---

# Update 2: match the original GG toolchain first

## Verified on Enzo's Mac (read-only check)
- Official GG data (HumanEval-C O0/O2, 328 rows): every row has `.build_version macos, 13, 0 sdk_version 13, 3`, i.e. Apple clang from Xcode 14.3 / macOS 13.3 SDK.
- Locally regenerated data (164 HumanEval rows, all of MBPP-C, all of ExeBench train/valid/test): `.build_version macos, 27, 0 sdk_version 27, 0`. Mac is macOS 27.0.1 with Command Line Tools only (Apple clang 21.0.0, SDK 27.0); no Xcode.
- Tart is not installed. `clang -arch arm64` assembles, links and runs a hello-world natively, so the reward-side basics work.
- Measured drift (old branch): regenerated vs official O2 prompts have 0% exact x86 match, ~55% similarity.

## Decision (Enzo): the reference toolchain defines the data
Everything that turns C into x86/ARM assembly (training prompts, references, test drivers) must come from a macOS 13.3 SDK / Xcode 14.3 toolchain. This is what the Tart VM is for: a macOS 13 guest with Xcode 14.3.x, running on the macOS 27 host.

## Revised order of work (no model training until step 3 passes)
1. **Reproduce the toolchain.** Create a Tart VM with macOS 13 + Xcode 14.3.x (Cirrus Labs publishes Ventura+Xcode images; exact tag to be verified before use). Inside it, check `clang --version` and that compiled output carries `.build_version macos, 13, 0 sdk_version 13, 3`.
2. **Acceptance test:** rerun `analysis/compiler_drift.py` inside the VM. Pass condition: regenerated HumanEval-C O0/O2 x86 and ARM match the official data (near 100% exact after stripping comments). If not, the version is still wrong; stop there.
3. **Regenerate datasets in that VM**, MBPP-C first (about 5 min class job), ExeBench later. Old regenerated parquet files are discarded; they were cheap. Keep official HumanEval-C O2 (164) untouched as the evaluation set so the paper number is comparable.
4. **Reward server inside the same VM** (harness and test drivers built with the matching toolchain), reached by one SSH tunnel to totoro.mbzu.ae.
5. **Baseline step-0 eval** on official HumanEval-C O2 (replicates ~45%), then the two TRL GRPO runs (LoRA, QLoRA) on the Blackwell GPU.

## Why this is reusable later
Any later arm, seed or dataset reuses the same pinned VM image and the same data build, so numbers stay comparable. Record in the repo: Tart image tag, `clang --version`, SDK version, and the `.build_version` line check.

---

# Update 3 (Oct 22): toolchain pinned, step 0 started

## Verified
- Tart VM `gg-ventura` (image `ghcr.io/cirruslabs/macos-ventura-xcode:14.3.1`): macOS 13.6, Xcode 14.3.1, Apple clang 14.0.3, SDK 13.3. Output carries `.build_version macos, 13, 0 sdk_version 13, 3`.
- Byte-for-byte vs GG's released assembly: HumanEval-C O0/O2 316/328 rows (6 problems differ in source: 0, 1, 101, 125, 137, 162), BringUp O0/O2 131/132 rows (only `satomi` O2 differs). Both x86 and ARM.
- Camera-ready paper (EMNLP Findings 2025): ARMv8 compiled natively on an Apple M2 Pro with clang; no versions given. Compile commands are in the repo (`train/compile_{arm,x86}_clang.py`): `clang -arch arm64 -S -O{level}` and `clang -S -target x86_64-apple-darwin -O{level}`.
- Released benchmark and SFT datasets contain assembly only (x86, arm, file path). No tests, no C source. Tests come from LLM4Decompile HumanEval-C and the BringUp Makefiles.

## Paper facts that matter
- HumanEval-C O2 45.12% and BringUp O2 15.38% (beam 8, no sampling, strict all-tests-pass pass@1). Ablation (Table 5, Fig. 5) is HumanEval O0.
- Training context 16k, inference RoPE-extrapolated to 32.7k; extended tokenizer. HumanEval mean ~1.2k tokens, BringUp ~11k. 17% of BringUp failures are context truncation.
- Post-training INT4 drops O0 HumanEval from ~98.2% to ~93.9% (read from Fig. 5).
- No decontamination statement against HumanEval/BringUp; no MBPP/ExeBench in training data.

## Order of work now
1. Step 0 (in progress via the Mac session): sanity-check harness in VM (reference ARM should pass ~158/164), generate with `gg-armv8-O2` beam 8 on totoro, score in VM, report 164 and 158-problem pass@1.
2. Authors' message (Xcode/clang version, extra flags, C source of 7 mismatching programs). Not blocking.
3. MBPP-C O2 built in the VM as RL pool (purely additive; see thread explanation). Held until step 0 reproduces ~45%.
4. TRL GRPO LoRA vs QLoRA on the Blackwell, W&B project `cs8001`.

## Pending questions
- Where `rl_simple/` code and VM outputs should be committed (nothing pushed yet).
- Does "the report" mean this plan file or midpoint_report.tex?
