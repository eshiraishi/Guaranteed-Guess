# Evidence: 02-harness (deep-research output, Oct 24, 2026)

Raw synthesis of the 3-vote verified claims. Fields: confidence | claim, SRC (sources), EV (evidence and vote). Sections after the findings: caveats, open questions, refuted claims, sources, stats.

SUMMARY: The strongest evidence says that decoupling generation from training (fully async or pipelined RL) cuts end-to-end time by about 1.5-2.3x against strong synchronous baselines at 1.5B-32B, with accuracy matched within noise; four independent primary sources agree on this (AReaL, PipelineRL, DeepCoder verl-pipeline, Periodic Asynchrony). In-flight weight updates into vLLM are the largest single throughput lever: in OlmoRL they took throughput from 1358 to 2949 tok/s, and the full stack ran 3.35x faster than the old OLMo 2 one on the same hardware. Overlapping the stages alone is capped at 2x, and reaches 2x only when generation and training take equal time. Gains beyond 2x come from scoring each rollout as soon as it finishes, which removes the wait for the slowest rollout in each batch, and staleness must be bounded (eta <= 4-8) with an off-policy-corrected objective or accuracy can collapse. Our current stack cannot take the cheap path: TRL 1.16's experimental AsyncGRPOTrainer does overlap generation and scoring with training, but it requires vLLM and the trainer on separate GPUs, FSDP2 and NCCL weight transfer, so our single RTX PRO 6000 colocate setup needs a second GPU or a different framework. On the reward side, published systems scale execution out behind a router with a global concurrency cap (ScaleBox, Olmo 3 on AWS Lambda, open-r1 routers and Piston fleets), which makes our single 8-worker macOS VM the outlier; moving rewards to Linux AArch64 also changes the target ABI (x18 reserved on Apple, variadic arguments on the stack, va_list as char*), and GG itself ran its ARMv8 tests natively on macOS.

- high | Async / decoupled RL (generation runs alongside training) cuts end-to-end training time by about 1.5-2.3x against strong synchronous baselines at 1.5B-32B, with final accuracy matched within noise. Headline figures above that (2.77x, 3.09x) are measured against weaker in-house or other-framework baselines.
  SRC: https://arxiv.org/html/2505.24298v4 https://arxiv.org/html/2509.19128v2 https://together.ai/blog/deepcoder https://arxiv.org/pdf/2511.18871
  EV: AReaL Table 1 (H800, 16-48 nodes, R1-distilled Qwen2):
- 1.5B math: 14.8 h async vs 41.0 h own-sync (2.77x) vs 33.6 h verl (2.27x). The verl hours are estimated from throughput, not measured. AIME24 42.2 vs 42.0.
- 7B: 2.05x vs verl.
- 14B code: 21.9 h vs 44.4 h verl (2.03x), LCB 58.1 vs 56.7 sync.
- 32B code: 31.1 h vs 51.1 h (1.64x vs sync, 1.49x vs verl), LCB 61.0 vs 61.2. Async scored slightly lower here.

The paper's own headline figures conflict: 2.77x, 2.57x and 'at most 2.5x' appear in different sections.

PipelineRL (Qwen2.5-7B, math, 128 H100): about 2x faster learning, driven by about 2x sample throughput. The baseline was a low-utilization G=32 conventional run whose 128-GPU throughput was extrapolated, and the paper gives inconsistent node counts (4 vs 16 nodes).

DeepCoder verl-pipeline (1.5B, 8 A100): one-off pipelining plus reward computed as each request completes cut end-to-end coding time 2x (1.4x for math), and up to 2.5x vs baseline verl. The baseline already parallelized rewards in a threadpool. The trade-off is one step off-policy.

Periodic Asynchrony (1.5B, GSM8K, 8x A100-40G):
- 1510 TPSPD (tokens trained per second per device) vs verl 489 (3.09x), AReaL 1068 (1.41x) and the authors' own sync 629 (2.4x).
- AReaL lost accuracy (0.681 vs 0.782); the async method held at 0.776.
- Configurations differed (3:1 vs 1:1 train-to-rollout ratio).

All results are self-reported, single-run, multi-GPU, mostly long-context reasoning.

- medium | Pipeline math: overlapping inference and training gives a step time of about max(T_infer, T_train). The speedup from overlap alone is therefore (T_infer+T_train)/max(T_infer, T_train) <= 2x, reached only when the stages are balanced. Gains above 2x come from enqueuing rollouts as they finish, which removes the slowest-rollout-per-batch barrier; otherwise the slower stage dominates. prime-rl's docs likewise say their default one-step overlap gives zero idle time only with matched step times, and they give no measured speedup.
  SRC: https://arxiv.org/pdf/2511.18871 https://docs.primeintellect.ai/prime-rl/async (moved to docs.primeintellect.ai/prime-rl/algorithms.md)
  EV: Periodic Asynchrony, Sec 4.2.2, Eqs. 3-4: Tsync/Tasync <= 2, with the bound approached when Tinfer ≈ Ttrain. Its own measurements:
- 1.92x over own sync for Qwen3-8B.
- 2.0x for Qwen2.5-7B with shared-prompt attention in both arms.
- 2.4x at 1.5B, consistent with the barrier-removal explanation.

prime-rl docs: 'With matched trainer and inference step times this produces fully-overlapped pipeline parallelism — neither side ever idles.'

Two verifier qualifications:
- The word 'only' (gains above 2x come only from barrier removal) is the claim's own strengthening; the paper does not say it.
- The bound assumes disaggregated inference and training. In that paper, reward scoring counts inside T_infer.

Inference for our harness: a separate slow reward server (the VM over an SSH tunnel) acts as a third stage. Overlap gains are bounded by whichever of generation, scoring and training is slowest, so we need the step-time breakdown before choosing a lever.

- high | In-flight weight updates are the largest single throughput lever in published async stacks. New weights are pushed into vLLM without draining or stopping generation, the KV cache is kept, and sequences end up mixed-policy.
  SRC: https://arxiv.org/pdf/2512.13961 https://arxiv.org/html/2509.19128v2
  EV: Olmo 3 / OlmoRL Table 23 (2x 8xA100 nodes, 1 learner + 1 inference, 2 h benchmark, 128 completions/step as 64 prompts x 2, 32K max output). Tokens per second:
- OLMo 2 stack: 881.
- + continuous batching: 975.
- + better threading: 1358.
- + in-flight updates: 2949. This step alone is 2.17x.

The total is 3.35x, computed from the table; the paper itself says 'up to 4x'. The caption calls in-flight updates 'the most drastic improvement'. Absolute MFU is about 1%, and Table 23 measures throughput only, not accuracy.

PipelineRL: 'the LLM generation engine only briefly pauses to receive the model weights' over the inter-accelerator network, then keeps decoding. Earlier tokens are more off-policy, the KV cache is kept stale (recomputing it is optional), and no pause duration is measured.

The batch size (128 completions/step) matches ours, but the rollout lengths (32K) are far longer than assembly completions.

- medium | Async training needs bounded staleness and an off-policy-corrected objective. With AReaL's decoupled PPO objective, staleness eta <= 8 barely moves final accuracy. Without it, accuracy can collapse. Unbounded staleness costs points even with the correction.
  SRC: https://arxiv.org/html/2505.24298v4 https://docs.primeintellect.ai/prime-rl/async (moved to /prime-rl/algorithms.md)
  EV: AReaL used eta=4 for coding and eta=8 for math. Table 2 (1.5B math, AIME24), without / with the decoupled objective:

| Staleness | Without | With |
|---|---|---|
| oracle eta=0 | 42.0 | 42.0 |
| eta=1 | 41.8 | 42.1 |
| eta=2 | 40.0 | 41.8 |
| eta=4 | 23.3 | 42.2 |
| eta=8 | 35.7 | 41.0 |
| eta=16 | 35.8 | 38.7 |
| eta=inf | 34.0 | 36.9 |

This is a single ablation (one model, math only, no seeds), and the uncorrected eta=4 result is non-monotonic, which points to one unstable run.

prime-rl runs one step overlapped by default. A claim that it runs with 'exactly one step of lag' was refuted 0-3, because its training docs expose orchestrator.max_off_policy_steps (default 8).

A claim about TRL AsyncGRPOTrainer defaults (max_staleness=4, weight_sync_steps=1, samples dropped past the limit) was refuted 0-3. TRL's actual staleness and correction behavior is therefore unverified.

- high | TRL 1.16 (the version our uv.lock pins) has an experimental AsyncGRPOTrainer. A background worker streams completions from a vLLM server and runs the reward functions while the trainer runs gradient steps. Its requirements rule it out on our single-GPU colocate setup: vLLM and the trainer on separate GPUs, vLLM dev mode with NCCL weight transfer, FSDP2 only (no DeepSpeed ZeRO), and transformers>=5.2.0. No async-vs-sync speedup is published.
  SRC: https://huggingface.co/docs/trl/async_grpo_trainer
  EV: The v1.16.0 docs say:
- 'A background worker continuously streams completions from a vLLM server while the training loop consumes them'.
- The rollout worker 'scores completions with reward functions, computes advantages', in a separate process, so reward functions must be picklable.
- 'Even in vLLM colocate mode ... one phase must finish before the other begins' (this is about the standard GRPOTrainer).
- 'The vLLM server and the trainer must run on **separate GPUs**'.
- 'only FSDP2 is supported (DeepSpeed ZeRO is not)'.
- The example server uses VLLM_SERVER_DEV_MODE=1 and --weight-transfer-config '{backend:nccl}'.

AsyncGRPOConfig has no colocate option. The main-branch docs match.

The page gives no speedup benchmark, and the HF blog 'Keep the Tokens Flowing' confirms that GRPOTrainer is synchronous.

Secondary note from the verifier: splitting the RTX PRO 6000 with MIG is unlikely to work, because NCCL does not support MIG and CUDA IPC does not cross MIG instances.

The wheel was uploaded 2026-10-15 and is marked experimental, so the API may change.

- medium | In long-rollout RL, generation dominates compute. Published stacks give inference roughly 5-14x the training compute, and the learner mostly waits for data. Short-completion tasks like ours will balance differently, so the split must be measured, not assumed.
  SRC: https://arxiv.org/pdf/2512.13961
  EV: OlmoRL ran with max 32K tokens per rollout and a mean above 10K.
- In a representative step of about 1000 s, only 125 s was training: 'the bottleneck is entirely generation'.
- 32B: 20 inference vs 8 training H100 nodes, 'learner spends 75% of the time waiting', described as about 5x inference compute.
- 7B: 7 inference vs 2 learner nodes, about 14x.
- Elsewhere the paper calls a typical run 8:1 inference to training nodes (or more).

The paper's own numbers are inconsistent: 125/1000 implies about 87.5% waiting, not 75%. The 1000 s step may not come from the 32B run.

The setup is fully async, so a sync TRL colocate harness would show different ratios. With short assembly completions, training and reward scoring are likely a larger share; this is an inference, not measured.

- medium | Measured sandbox throughput for Python code RL: one 64-core ScaleBox node grades about 39 tasks/s, 1.59x verl's native Prime executor and 2.63x SandboxFusion. Scaling across nodes behind NGINX round-robin is clearly sub-linear.
  SRC: https://arxiv.org/html/2604.27467
  EV: ScaleBox Table 1 (8,192 PrimeIntellect Python problems, Intel Xeon 6700P, 64 cores per node):
- verl Prime: 24.73 tasks/s (331.25 s).
- SandboxFusion: 14.92 tasks/s (548.93 s).
- ScaleBox, 1 node: 39.31 tasks/s (208.38 s).
- ScaleBox, 2 nodes: 50.13 tasks/s (1.28x vs 1 node, ideal 2x).
- ScaleBox, 3 nodes: 62.10 tasks/s (1.58x, ideal 3x).

That is about 0.6 tasks/s per core on one node. The paper says 'substantial scalability'; 'sub-linear' is the verifiers' reading of its numbers.

Caveats: the authors benchmarked their own system, Python only, in a single run with no error bars, published April 2026 (ACL 2026 demo).

Inference for us: at that rate, one 128-completion step would take about 3 s of grading on one such node. Assemble+link+run per test, plus a VM/tunnel round trip, will cost more per task; this is unmeasured.

- medium | Published code-RL harnesses all scale execution out behind a central dispatcher that enforces concurrency limits, and score per completion as results arrive. Our single VM reached over an SSH tunnel is the outlier.
  SRC: https://arxiv.org/pdf/2512.13961 https://github.com/huggingface/open-r1 https://together.ai/blog/deepcoder
  EV: Olmo 3 runs RL code rewards on AWS Lambda.
- The base-eval appendix gives the figures: capacity 'up to 50K generations simultaneously', 17.2M samples graded during development, peak 1.5K concurrent.
- A local Docker library (allenai/olmes-docker) runs without AWS.
- The paper gives no tests/s or latency numbers.

open-r1 uses hosted E2B (Python) or Morph (Python/JS/C++/Rust) sandboxes.
- The README warns 'It is possible to be rate limited'.
- Router scripts run on a CPU node, and all training jobs share one router IP. The router caps concurrency with an asyncio.Semaphore; Morph's default is 20 sandboxes.

The Codeforces judge is a self-hosted Piston fleet, one Slurm job (or local Docker container) per worker.
- PISTON_MAX_REQUESTS_PER_ENDPOINT defaults to 1 per worker.
- There is 'no global limit, workers might sometimes be overwhelmed'.
- IOI can also run on Morph.

DeepCoder computes each reward 'as soon as a request completes'.

A claim that DeepCoder needed 100+ concurrent Together sandboxes at 1k+ executions per minute was refuted 1-2.

Implication: the design to copy is a router with a global concurrency limit in front of N worker hosts, scoring each completion as it streams in. A per-endpoint limit alone overloads workers.

- medium | open-r1's code reward is graded (fraction of test cases passed), not binary pass/fail. A binary variant exists only as a 0.99 threshold over the graded score.
  SRC: https://github.com/huggingface/open-r1
  EV: README: 'solutions are executed against a set of test cases and the overall success rate is returned as the final reward'.

In src/open_r1/rewards.py:
- success_rate = passed/total.
- Each test runs in a subprocess with a 5 s timeout, and a nonzero exit counts as failure.
- binary_code returns 1.0 only when the score is above 0.99.
- cf_code_reward defaults to scoring_mode weighted_sum.

The repo is marked no longer maintained (TRL is recommended instead). Its code reward runs only Python, even on Morph.

This contrasts with our binary verdict per completion. Whether partial credit helps assembly RL is not addressed by any verified claim.

- high | Apple arm64 departs from Linux AArch64 at the assembly level in ways that change reward verdicts:
- x18 is reserved and must not be used.
- Every variadic argument goes in 8-byte stack slots, not registers.
- va_list is char*, not the AAPCS64 struct.

So a Linux AArch64 reward environment (Graviton, Ampere, QEMU) is a different target, not a drop-in replacement for the macOS VM.
  SRC: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
  EV: Apple doc: 'The platforms reserve register x18. Don't use this register.' Also: 'Assign the variadic argument to the appropriate number of 8-byte stack slots ... va_list is an alias for char*, and not for the struct type in the generic procedure call standard.' And: hand-written assembly 'must adhere to these rules ... may behave unexpectedly or even crash.'

The verifier cross-checked AAPCS64 (ARM abi-aa). r18 is 'the Platform Register, if needed; otherwise a Caller-saved register'. Linux leaves it free except under ShadowCallStack. AAPCS64 itself advises portable hand-written assembly to avoid x18.

Consequences:
- x18-using code can pass on Linux and be invalid on macOS.
- printf-style call setup differs: Linux passes variadic arguments in x0-x7/v0-v7.

Options: fix the ABI per reward environment and label data by target, or reject x18 on every target with a static check.

The underscore symbol prefix and other Mach-O vs ELF directive differences were not covered by a verified claim.

- high | GG compiled and ran its ARMv8 tests natively on an Apple M2 Pro under macOS with clang. It cross-compiled ARMv5 and RISC-V64 on an x86 Ubuntu 20.04 machine with gcc cross toolchains. The paper never says how or where the RISC-V64 tests were executed, and QEMU appears only as background.
  SRC: https://arxiv.org/html/2506.14606v1 /Users/eshiraishi/git/Guaranteed-Guess-feature-env/2025.findings-emnlp.1330.pdf
  EV: Sec 3.1, identical in arXiv v1 and the EMNLP 2025 Findings PDF: 'Compilation for ARMv5 and RISC-V64 was performed via cross-compilation on an Ubuntu 20.04 machine with a Ryzen 7 CPU, using arm-linux-gnueabi-gcc ... and gcc-riscv64-linux-gnu ... ARMv8 binaries were compiled natively on an Apple M2 Pro (macOS) using clang.'

Fig 1 labels the test platform MAC, and Sec 5.2 cites native ARM toolchain support on the M2 Pro.

QEMU appears only in the Intro, the Background section and the references. The paper reports 89.93% RISC-V64 test accuracy without naming an emulator or board, and the repo's eval/, train/ and README.md never mention qemu.

Implication: GG's AArch64 numbers use the Apple ABI, so comparisons with GG should keep a macOS reward target. For RISC-V64 we must choose the execution method ourselves, with no GG precedent.

== caveats
"How far the async results transfer: every speedup number is self-reported by the system's authors, comes from single runs without error bars, and was measured at 8-384 GPUs, mostly on long-context (10K-32K token) math or code reasoning. None was measured on one GPU with short assembly completions and a remote reward server. Several key baselines are estimated, not measured: AReaL's verl training hours come from throughput measurements, and PipelineRL's conventional-RL throughput is extrapolated. Several papers contradict themselves on their own numbers: AReaL gives 2.77x, 2.57x and 'at most 2.5x' in different sections; OlmoRL's 75% learner wait does not match 125 s of training in a 1000 s step; PipelineRL gives two different node counts. The Periodic Asynchrony comparison uses different train-to-rollout ratios per framework.\n\nTime sensitivity:\n- TRL AsyncGRPOTrainer is marked experimental, and its 1.16.0 wheel was uploaded 2026-10-15, nine days ago.\n- The cited prime-rl docs URL now returns 404; the text moved to /prime-rl/algorithms.md.\n- DeepCoder's verl-pipeline is about 18 months old and probably superseded by upstream async support in verl.\n- open-r1 is unmaintained.\n\nRefuted claims, which should not be relied on:\n- TRL AsyncGRPOTrainer staleness defaults (0-3).\n- prime-rl 'exactly one step' of policy lag (0-3); max_off_policy_steps defaults to 8.\n- DeepCoder needing 100+ sandboxes at 1k+ executions per minute (1-2).\n\nCoverage gaps: several research questions have no verified claim at all from this run.\n- Sandboxes: Firecracker, gVisor, nsjail, bubblewrap, SandboxFusion internals, caching and dedup of identical completions.\n- Running the target binaries: QEMU user-mode speed and fidelity, Graviton/Ampere, Virtualization.framework, the macOS 2-VM EULA limit, the Mach-O underscore prefix.\n- Reward hacking and sandbox hardening, and formal or symbolic equivalence checking.\n- CPU cores per GPU in DeepCoder, Skywork-OR1, AceReason and rStar.\n\nFive claims for this topic failed verification and are still pending. Several verifiers ran out of web-search budget, so searches for contradicting sources were incomplete for claims [12], [15], [19], [20] and [21]. Statements about our own harness (third reward stage, about 3 s per step of grading on a ScaleBox-class node) are inferences, not measurements."
== openQuestions
[
 "How do our 4-6 min steps actually divide between vLLM generation, reward scoring (VM plus SSH tunnel round trip) and the training step? That split decides whether async overlap (capped at 2x) or scaling out the reward workers is the bigger lever.",
 "Can a Linux AArch64 reward pool (Graviton, Ampere, or QEMU user-mode on x86) replace or add to the macOS VMs without changing reward verdicts, given x18, variadic and Mach-O symbol differences? What are QEMU user-mode's measured speed and fidelity for assemble+link+run on AArch64 and RISC-V64?",
 "Do bounded-staleness results (eta <= 4-8, decoupled PPO) hold for short-completion assembly RL with binary rewards? Does TRL's AsyncGRPOTrainer apply an off-policy correction comparable to AReaL's, or only drop samples older than the limit?",
 "Which defenses against reward hacking does an assembly harness need (stub outputs, tampering with the test driver, exit-code tricks, syscalls), and what does each cost in throughput? Options include sandbox-exec vs nsjail/bubblewrap/gVisor, differential checks against the reference, and SMT/angr-style equivalence checks. How many CPU cores per GPU do published code-RL recipes provision?"
]
== refuted
[
 {
  "claim": "Off-policyness is limited by the number of policy versions a sample lags, not by an importance-sampling correction. max_staleness defaults to 4 and weight_sync_steps defaults to 1, so weights go to vLLM over NCCL after every training step. Samples older than max_staleness are thrown away. By default, in-flight requests are capped at max_staleness x per_device_train_batch_size x grad_accum x num_processes.",
  "vote": "0-3",
  "source": "https://huggingface.co/docs/trl/async_grpo_trainer"
 },
 {
  "claim": "prime-rl runs async by default with exactly one step of policy lag. While the trainer computes policy n from step-n rollouts, inference is already generating the step n+1 rollouts with policy n-1. So rollouts are at most one policy version stale.",
  "vote": "0-3",
  "source": "https://docs.primeintellect.ai/prime-rl/async"
 },
 {
  "claim": "DeepCoder's code-RL reward stage needed 100+ parallel sandboxes; the Together Code Interpreter was scaled to 100+ concurrent sandboxes and 1k+ executions per minute at about 3 cents per problem, alongside a local guard-railed Python subprocess sandbox.",
  "vote": "1-2",
  "source": "https://together.ai/blog/deepcoder"
 }
]
== unverified
[]
== sources
[
 {
  "url": "https://arxiv.org/html/2505.24298v4",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://huggingface.co/docs/trl/async_grpo_trainer",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2509.19128v2",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2512.13961",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://docs.primeintellect.ai/prime-rl/async",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2511.18871",
  "quality": "primary",
  "angle": "Async RL frameworks: throughput benchmarks",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2604.27467",
  "quality": "primary",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://together.ai/blog/deepcoder",
  "quality": "primary",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://github.com/huggingface/open-r1",
  "quality": "primary",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://e2b.dev/blog/how-hugging-face-is-using-e2b-to-replicate-deepseek-r1",
  "quality": "blog",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://verl.readthedocs.io/en/latest/_sources/sglang_multiturn/sandbox_fusion.rst.txt",
  "quality": "primary",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://www.beam.cloud/blog/sandbox-code-execution-rlvr",
  "quality": "blog",
  "angle": "Code-execution sandboxes at RL scale",
  "claimCount": 5
 },
 {
  "url": "https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms",
  "quality": "primary",
  "angle": "AArch64/RISC-V execution and Apple arm64 ABI",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2506.14606v1",
  "quality": "primary",
  "angle": "AArch64/RISC-V execution and Apple arm64 ABI",
  "claimCount": 5
 },
 {
  "url": "https://mail.openjdk.org/pipermail/core-libs-dev/2021-May/077025.html",
  "quality": "primary",
  "angle": "AArch64/RISC-V execution and Apple arm64 ABI",
  "claimCount": 4
 },
 {
  "url": "https://arxiv.org/pdf/2606.16062",
  "quality": "primary",
  "angle": "Reward hacking and equivalence checking",
  "claimCount": 5
 },
 {
  "url": "https://icml.cc/virtual/2026/poster/63289",
  "quality": "primary",
  "angle": "Reward hacking and equivalence checking",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2510.20270",
  "quality": "primary",
  "angle": "Reward hacking and equivalence checking",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2309.14396",
  "quality": "primary",
  "angle": "Reward hacking and equivalence checking",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2505.16400",
  "quality": "primary",
  "angle": "Practitioner throughput math: CPU cores per GPU",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2509.21009v1",
  "quality": "primary",
  "angle": "Practitioner throughput math: CPU cores per GPU",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2505.22312v2",
  "quality": "primary",
  "angle": "Practitioner throughput math: CPU cores per GPU",
  "claimCount": 5
 }
]
== stats
{
 "angles": 5,
 "sourcesFetched": 22,
 "claimsExtracted": 109,
 "claimsVerified": 25,
 "confirmed": 22,
 "killed": 3,
 "unverified": 0,
 "afterSynthesis": 11,
 "urlDupes": 1,
 "budgetDropped": 7,
 "agentCalls": 104
}
