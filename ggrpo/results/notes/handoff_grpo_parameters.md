# Handoff: choosing the GRPO training parameters (ggrpo)

Written Oct 22, 2026 by session `guaranteed-guess-feature-env-c9` for a new thread whose only job is to help Enzo decide
the training parameters. Facts are marked **verified** (measured or read in source), **paper** (stated in the GG paper),
or **open** (unknown).

## Your scope (read first)
- Help Enzo decide every `TODO` in `ggrpo/grpo.yaml`, plus any setting he wants to revisit. Propose values with a
  source for each: the paper, the data below, TRL docs or source, or a cited RL paper. If a value has no source, say so
  and leave it `TODO`. Enzo decides; do not pick silently.
- Edit `ggrpo/grpo.yaml` only after Enzo agrees to a value. Do not edit other repo files.
- Do not start GPU jobs. Do not touch ponyo, totoro, or the Tart VM. Do not message other Claude sessions: Enzo wants a
  single thread for runs and data, and that is `guaranteed-guess-feature-env-c9`. If a decision needs a measurement
  (e.g. base pass rate under sampling), write down exactly what to measure and Enzo will ask that thread.
- Enzo's style rules are in the auto-loaded memory `python-style-preferences`: library entry points over custom code,
  no invented values, TOML for our settings and native YAML for TRL, uv, no commits until agreed.

## Project in three lines
- Start from the released GG model `ahmedheakl/gg-armv8-O2` (Qwen2.5-Coder-1.5B, SFT'd for x86 → ARMv8 O2 assembly),
  revision `0444a378f384300187c7f43914fc986a81ff051a`. Train it with GRPO through TRL using a binary test reward, and
  compare LoRA (bf16) against QLoRA (NF4).
- Reward: 1.0 if the completion's ARM assembly assembles, links with the task's C test driver, and passes it inside a
  macOS 13 VM with GG's exact toolchain, else 0.0. Trainer: TRL's own CLI, `trl grpo --config grpo.yaml`, with no
  train.py. QLoRA is the same file plus `--load_in_4bit true --bnb_4bit_quant_type nf4`.
- Evaluation: HumanEval-C O2 (164 problems) with the authors' own tests. BringUp O2 (65 programs) is secondary.

## Files to read (worktree `/Users/eshiraishi/git/Guaranteed-Guess-feature-env/ggrpo`)
- `grpo.yaml`: the TRL config with the TODOs. `config.toml`: our settings (prompt template, extraction regex, server,
  harness limits, `[generate.sample]` decoding that must match grpo.yaml).
- `src/ggrpo/reward.py` (TRL reward function → HTTP reward server), `src/ggrpo/server.py`, `src/ggrpo/harness.py`,
  `src/ggrpo/data.py`.
- `docs/code_plan.md`, `docs/status-and-plan.md`. `docs/plan-history.md` is older and partly superseded.
- Paper: "Guaranteed Guess", EMNLP Findings 2025, arXiv 2506.14606. Key facts are below.

## Current grpo.yaml (verified Oct 22, 15:35)
Fixed already: `num_generations: 8`, `beta: 0.0`, `mask_truncated_completions: true`, `use_peft: true`,
`lora_target_modules: all-linear`, sampling `temperature 1.0, top_k 0, top_p 1.0, repetition_penalty 1.0`,
`use_vllm: true`, `vllm_mode: colocate`, `bf16: true`, `gradient_checkpointing: true`, `seed: 0`, Trackio local
(`project: cs8001`), `save_steps: 25`, `logging_steps: 1`, `log_completions: true`.
TODO: `lora_r`, `lora_alpha`, `lora_dropout`, `loss_type`, `learning_rate`, `max_steps`,
`per_device_train_batch_size`, `gradient_accumulation_steps`, `max_completion_length`.
Worth revisiting even though set: the sampling temperature (see "Decisions" 6), and whether vLLM colocate is right for
QLoRA (see "Risks" 1).

## TRL 1.16.0 defaults (verified in installed source, trl/trainer/grpo_config.py and model_config.py)
`learning_rate 1e-6`, `loss_type "dapo"` (options: grpo, dr_grpo, dapo, bnpo, cispo, sapo, luspo, vespo; TRL's help
text says plain "grpo" has a length bias), `epsilon 0.2`, `epsilon_high None`, `beta 0.0`, `num_generations 8`,
`max_completion_length 512`, `temperature 1.0`, `top_p 1.0`, `top_k 0`, `repetition_penalty 1.0`,
`scale_rewards "group"`, `num_iterations 1`, `mask_truncated_completions False`, `vllm_gpu_memory_utilization 0.3`,
`importance_sampling_level "token"`, `vllm_importance_sampling_mode "sequence_mask"` (vLLM/trainer mismatch correction
is on by default). LoRA (ModelConfig): `lora_r 16`, `lora_alpha 32`, `lora_dropout 0.05`.
Generation batch: `steps_per_generation` defaults to `gradient_accumulation_steps`. The generation batch
`per_device_train_batch_size × processes × steps_per_generation` must be divisible by `num_generations`.
Reward kwargs TRL passes: dataset columns (only `task_id` here), `trainer_state`, `log_extra`, `log_metric`.

## Library versions (uv.lock, verified)
trl 1.16.0, transformers 5.17.0, torch 2.13.0 (CUDA 13 wheels), vllm 0.31.0, peft 0.21.2, bitsandbytes 0.50.2,
trackio 0.43.0. TRL 1.16 supports vLLM ≤ 0.31.0.

## Data (verified)
- Train: MBPP-C O2, 952 tasks (reference ARM passes, empty prediction fails), compiled in the VM with GG's toolchain.
  `data.train_ids` can restrict it to a vetted subset; unset by default.
- Token lengths (model tokenizer, chat template):
  - Train prompts: p50 930, p90 2,552, p99 5,006, max 10,749.
  - Train reference completions (fenced ARM): p50 717, p90 2,068, p99 4,102, max 7,546. 97 tasks exceed 2,048 tokens;
    11 exceed 4,096.
  - Eval (HumanEval) prompts: p50 1,007, max 6,405. Eval reference completions: p50 811, p99 3,759, max 4,169.
- Base-model outputs (beam 8, 16k cap): new tokens p50 797, p90 1,929, p95 3,194. 7 of 164 ran away to the cap and all
  failed. Runaways matter for `max_completion_length` and the loss choice.

## Baseline and evaluation (verified, Oct 22)
- Toolchain: macOS 13.6 VM, Xcode 14.3.1, clang 14.0.3, SDK 13.3. It reproduces GG's released assembly byte for byte.
- HumanEval-C O2, base model, beam 8, no sampling: **72/164 (43.9%)** with the authors' tests; paper **74/164 (45.12%)**.
  Runs that test whether decoding or library versions explain the 2-problem gap are in progress in the other thread.
- Agreed direction (Enzo to confirm): main metric is beam 8 without sampling on all 164 problems, the same protocol
  for base, LoRA and QLoRA. Sampled pass@1 at the training temperature is secondary.
- The released checkpoint's generation_config.json sets do_sample true, T 0.7, top_k 20, top_p 0.8,
  repetition_penalty 1.1. TRL does not use it (TRL passes explicit sampling values), and grpo.yaml currently overrides
  to T 1.0 with no top-k/top-p and no penalty.

## Paper facts useful as sources (paper)
- SFT: Qwen2.5-Coder-1.5B, about 1.31M samples (AnghaBench 1M + Stack v2 300k), O0 and O2, 2 epochs, lr 2e-5 with a
  cosine schedule, paged AdamW, weight decay 0.001, bf16, 16k training context, RoPE extrapolation to 32.7k at inference.
  The text says batch size 24 with an effective batch size of 2 (likely a typo). A100 40 GB, 3 days.
- Inference: beam 8, pass@1 with all tests required. The GG paper itself has no RL.
- The paper's ablation shows context length matters most: 4k → 16k is worth about +10 points at O0.

## Hardware (verified)
- totoro: NVIDIA RTX PRO 6000 Blackwell, 96 GB, sm_120, needs torch ≥ 2.7. Second card: AD102, 16 GB. Currently
  down (driver/library mismatch) and waiting for a reboot. When it comes back, BringUp evaluation runs (two processes,
  roughly 20–40 GB) will share the Blackwell for several hours.
- ponyo: RTX 4070 Ti SUPER, 16 GB; busy with HumanEval reproduction runs for many hours.
- Reward server: inside the macOS 13 VM on Enzo's Mac (4 vCPUs, 8 worker processes). Each check takes about 0.1–0.5 s
  (assemble, link, run with a 10 s wall and 5 s CPU limit). It is reached from totoro through one SSH reverse tunnel.

## Decisions to make (open)
1. **loss_type**: grpo / dapo (TRL default) / dr_grpo / others. Length bias matters because some completions run
   away. Related: `epsilon_high` (DAPO's clip-higher), `mask_truncated_completions` (already true).
2. **LoRA r / alpha / dropout** (TRL defaults 16 / 32 / 0.05). The same values for LoRA and QLoRA keep the comparison
   clean.
3. **learning_rate** (TRL default 1e-6). An older internal note says TRL docs suggest about 1e-5 for LoRA GRPO; verify
   that in the current docs before citing it.
4. **Batch geometry**: `per_device_train_batch_size`, `gradient_accumulation_steps`, `steps_per_generation`. These set
   prompts per step (each × 8 completions) and therefore reward-server load per step.
5. **max_completion_length**: the data above shows the trade-off. 2,048 truncates the references of 97 training tasks;
   4,096 truncates 11. Truncated completions are masked from the loss.
6. **Sampling for rollouts**: currently T 1.0 with no top-k/top-p. The model was SFT'd and is used at T 0.7 / top_p 0.8
   by default. The rollout temperature sets how often a group of 8 has mixed rewards (the only groups with a GRPO
   signal). Open measurement: the fraction of training tasks with 1–7 passes out of 8 at the chosen temperature.
7. **max_steps / epochs** over 952 tasks, and the checkpoint cadence (`save_steps 25`).
8. **QLoRA specifics**: NF4, double quantization or not, compute dtype bf16, and whether LoRA and QLoRA use exactly the
   same hyperparameters.
9. **Training subset**: all 952 tasks, or a filtered subset (e.g. drop tasks the base model always passes or always
   fails; that needs the measurement in 6).
10. **Contamination check** (open): MBPP-C overlap with GG's public SFT data was never measured.

## Risks already identified (verified in TRL 1.16 source, not yet run)
1. QLoRA + vLLM colocate: TRL starts vLLM with `quantization="bitsandbytes"`, and at every weight sync it merges the
   LoRA adapter into the 4-bit base, then unmerges it (trl/generation/vllm_generation.py). This can accumulate rounding
   drift in the frozen base, and it is untested whether vLLM accepts those 4-bit tensors. A short smoke test must pass
   before any QLoRA run; `use_vllm: false` is the fallback (slower).
2. vLLM loads the model by name without the pinned revision (`model.name_or_path`). Hugging Face `main` currently
   equals the pinned revision.
3. No `max_prompt_length` in TRL 1.16's GRPOConfig. Training prompts reach 10,749 tokens, so memory planning must
   include prompt + completion.
4. A reward server failure (HTTP 404/503) stops training on purpose; it never counts as a zero reward.
