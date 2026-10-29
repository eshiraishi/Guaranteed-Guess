# GG GRPO: Results So Far

Oct 23, 2026 · Enzo Shiraishi

One run of TRL's default GRPO configuration on the released GG model (`ahmedheakl/gg-armv8-O2`, 1.5B) raised HumanEval-C O2 from 72 to 87 of 164 problems (beam 8). The held-out dev set stayed flat. A second seed, two named baselines (DeepSeekMath GRPO and Dr. GRPO) and the trained-model BringUp scores are still running.

## Setup

We train the released GG checkpoint with TRL's default GRPO configuration, called **GRPO (TRL default)** below, using a reward of 1 when its ARMv8 assembly passes all tests, and evaluate it with the GG paper's protocol. GRPO (TRL default) is not the original DeepSeekMath GRPO; Baseline arms lists the differences.

| Item | Value |
| --- | --- |
| Starting model | `ahmedheakl/gg-armv8-O2` @ 0444a378 (Qwen2.5-Coder-1.5B after GG's full-parameter SFT, x86 to ARMv8 at O2) |
| Training data | MBPP-C O2: MBPP problems in C with their asserts as tests (from `LLM4Binary/decompile-eval`), compiled with GG's commands. 952 valid tasks: 858 train, 94 dev |
| Reward | 1 if the generated assembly assembles, links with the C test driver and passes every assert; else 0 |
| Reward harness | macOS 13.6 VM on Apple Silicon (Xcode 14.3.1, Apple clang 14.0.3, SDK 13.3), called over an SSH tunnel |
| Trainer | TRL 1.16 GRPO, full-parameter, fp32 master weights with bf16 autocast, vLLM generation (colocated) |
| Hyperparameters | lr 1e-6 decaying linearly to 0, no warmup (the transformers default schedule); token-level loss (TRL `loss_type: dapo`); no KL penalty (beta 0); 8 completions per prompt, 16 prompts per step, one optimizer step per rollout batch; max 4,096 completion tokens (truncated ones masked); temperature 1 |
| Length | 106 steps = 2 epochs; checkpoints at 27, 54, 81 and 106 |
| Reported model | The final checkpoint (106), fixed before training; dev is never used to pick a checkpoint |
| Training hardware | 1x RTX PRO 6000 Blackwell (96 GB); 6.5 h per run |
| Main metric | pass@1 with beam search, 8 beams, as in the GG paper: a problem counts only if all its tests pass |
| Eval sets | HumanEval-C O2 (164 problems, the GG authors' own tests) and BringUp O2 (65 whole programs); both held out from training |
| Eval hardware | HumanEval and dev on RTX 4070 Ti SUPER; BringUp on RTX 5000 Ada (needs transformers 4.50). Base and trained models always run on the same GPU type and software |

## Base GG model

Our runs of the released GG-1.5B model score 72 to 77 of 164 on HumanEval-C O2, depending on software versions; the paper reports 74. On BringUp we get 6 of 65 against the paper's 10.

**GG paper (EMNLP Findings 2025, Table 3, ARMv8).**

| Model | HumanEval-C O0 | HumanEval-C O2 | BringUp O0 | BringUp O2 |
| --- | --- | --- | --- | --- |
| GG-1.5B | 99.39% | 45.12% (74/164) | 49.23% | 15.38% (10/65) |
| GG-0.5B | 86.06% | 25.45% | 27.69% | 3.08% |
| GPT-4o | 10.3% | 4.24% |  |  |

**Our runs of GG-1.5B at O2.**

| Setup | HumanEval-C O2 |
| --- | --- |
| Our training stack (transformers 5.17, torch 2.13): beam 8, no sampling, repetition penalty 1.1. **The base-model reference for every trained-model comparison** | 72/164 (43.9%) |
| The paper's software (transformers 4.50, torch 2.6, FlashAttention 2): beam 8, no sampling | 77/164 (47.0%) |
| The authors' released eval script, unchanged: it inherits sampling (temperature 0.7, top-k 20, top-p 0.8) from the model's config. Seeds 42 and 1 give 160/164 identical outputs | 76/164 (46.3%) |
| Sampled pass@1 (temperature 1, 8 samples per problem, our stack) | 31.3% |

| Setup (RTX 5000 Ada, transformers 4.50, torch 2.7.1) | BringUp O2, all 65 | BringUp O2, 53 clean |
| --- | --- | --- |
| Beam 8 | 6/65 (9.2%) | 6/53 |
| Greedy, the authors' released BringUp script | 5/65 (7.7%) | 5/53 |

- BringUp passes at beam 8: hanoi, indirect-test, quine, skeleton, strange, weekday. Greedy loses hanoi.
- At beam 8, 6 BringUp programs do not fit the context window and count as failures.
- "Clean" excludes 12 BringUp programs that appear in GG's SFT data (see Harness and data checks).
- Software versions alone move HumanEval-C by about 5 problems (3 points), with identical decoding settings. We therefore compare base and trained models on one stack only.

## GRPO (TRL default) results, seed 0

The final checkpoint solves 15 more HumanEval-C problems than the base model (87 vs 72 of 164, beam 8). Beam-8 accuracy on dev did not move.

**HumanEval-C O2, beam 8 (same GPU and software for both rows).**

| Model | Solved | Pass | Crash | Assemble error | Timeout |
| --- | --- | --- | --- | --- | --- |
| Base GG-1.5B | 72/164 (43.9%) | 72 | 75 | 16 | 1 |
| GRPO (TRL default), checkpoint 106 | 87/164 (53.0%) | 87 | 61 | 14 | 2 |

- Per problem: 16 gained, 1 lost (humaneval/120); exact sign test p = 0.0003. This tests variation across problems within one run, not across training seeds.
- 91 of 164 outputs are identical to the base model's. The median output length is unchanged (797 tokens).
- Without the 12 HumanEval problems that have a twin in MBPP-C: 65 → 79 of 152.

**Dev (94 held-out MBPP-C tasks) across checkpoints.**

| Checkpoint (step) | Beam 8 | Sampled pass@1 (T = 1, 8 samples) | Sampled pass@8 |
| --- | --- | --- | --- |
| Base | 56/94 | 45.1% | 62/94 |
| 27 | 53/94 | 46.7% | 63/94 |
| 54 | 53/94 | 47.5% | 60/94 |
| 81 | 55/94 | 48.8% | 63/94 |
| 106 | 54/94 | 48.3% | 60/94 |

- Sampled pass@1 rises steadily: at 106, 22 tasks improve and 8 get worse (sign test p = 0.016). pass@8 is flat, so training makes correct answers more likely on tasks the model could already solve; it does not add new ones.
- Sampled pass@1 is a diagnostic only, not a reported metric. It matches the decoding used during training.
- Dev is flat under beam 8 while HumanEval-C gains 15. We do not have an explanation yet.

**BringUp O2.** Beam-8 generation for checkpoint 106 is complete and waits for scoring; the greedy run is finishing. See Pending runs.

## Training dynamics, seed 0

The training pass rate rose from 45.5% in epoch 1 to 48.4% in epoch 2. More than half of the prompt groups gave no learning signal throughout.

| Logged metric (mean per step) | Epoch 1 (steps 1–53) | Epoch 2 (steps 54–106) |
| --- | --- | --- |
| Reward = share of completions that pass | 0.455 | 0.484 |
| Prompt groups with all-pass or all-fail (no gradient) | 57.0% | 57.7% |
| Policy entropy | 0.0355 | 0.0300 |
| Mean completion length (tokens) | 969 | 962 |
| Completions cut at 4,096 tokens | 1.5% | 1.3% |
| Crash | 37.2% | 35.9% |
| Assemble error | 14.2% | 12.6% |
| Timeout | 2.9% | 3.0% |
| Seconds per step | 197 | 195 |

- The first 10 steps average 0.397 reward and the last 10 average 0.487. Prompts differ in difficulty from step to step, so compare by epoch, not by single steps.
- Entropy is already low at the start (the model was SFT'd) and drops by about 15%.
- The 57% of groups with no gradient points to dynamic sampling or a harder prompt mix as next levers.
- At bf16 precision, only about 6% of the weights differ from the base model at step 27 (lr 1e-6).
- The learning rate decays linearly from 1e-6 at step 1 to about 0 at step 106 (5e-7 at step 54). An earlier config comment called it constant; the logs show it was not.
- PPO clip ratios are exactly 0 at every step. Each rollout batch gets one optimizer step, so the policy ratio is always 1 and clipping never acts.

## Charts we can make

Training curves come straight from Trackio, which logs every step. Eval charts come from the scored result files, because evals run offline after training.

Trackio (W&B-compatible API), project `cs8001`. Runs: `baseline, seed=0` (done) and `baseline, seed=1` (paused at step 27, resumes after the baseline arms) are GRPO (TRL default); `pilot` is a 952-prompt run that ran out of memory at step 19. The two baseline arms will appear as new runs. Any chart below can overlay seeds and arms.

**From Trackio, per training step.**

| Chart | Logged keys | What it shows |
| --- | --- | --- |
| Reward curve | `train/reward`, `train/rewards/reward_pass/mean` and `/std` | Training pass rate over steps |
| Failure modes | `train/verdict/{pass, crash, assemble_err, timeout, wrong_output, link_err, empty, infra_err}` | Which errors RL removes (stacked area) |
| Zero-signal groups | `train/frac_reward_zero_std` | Share of prompts where all 8 completions pass or all fail |
| Entropy | `train/entropy` | Exploration left in the policy |
| Completion length | `train/completions/{mean, min, max}_length`, `mean_terminated_length`, `clipped_ratio` | Length drift and how often outputs hit 4,096 tokens |
| Optimizer | `train/loss`, `train/grad_norm`, `train/learning_rate` | Training stability; the linear LR decay |
| vLLM vs trainer agreement | `train/sampling/importance_sampling_ratio/*`, `sampling_logp_difference/*` | How off-policy the generations are |
| Clipping | `train/clip_ratio/{low_mean, high_mean, region_mean}` | Always exactly 0 in our runs (one optimizer step per rollout batch), so not worth a chart |
| Cost | `train/step_time`, `train/num_tokens`, `profiling/*` (generation, reward, loss, weight sync) | Where the 6.5 h goes |
| System | GPU utilization, memory, power, temperature | Hardware load |
| Completions table | `completions` (prompt, completion, reward per step) | Example outputs for qualitative analysis |

**From the scored result files (not in Trackio yet; another thread is adding eval logging).**

- Dev learning curve: beam 8 and sampled pass@1 against step (base, 27, 54, 81, 106), one line per seed.
- HumanEval-C bars: paper 74, our base 72, GRPO (TRL default) 87, plus seed 1 and the two baseline arms when they land.
- Verdict mix on HumanEval-C, base vs trained (the table in GRPO (TRL default) results).
- Per-problem flips: which problems were gained and which lost, by seed.
- BringUp: base vs trained, on all 65 and on the 53 clean programs.
- Base-model reproduction: paper vs our software stacks (Base GG model tables).

## Harness and data checks

The harness reproduces GG's released data exactly, and the HumanEval-C gain does not come from contamination we can detect.

**Toolchain and tests.**

- Our macOS 13 VM recompiles GG's released HumanEval-C assembly byte for byte from the authors' C sources: all 328 rows (O0 and O2). BringUp: 131 of 132 rows (only `satomi` at O2 differs).
- With the authors' tests, the reference ARM assembly passes 164/164 HumanEval-C problems, and an empty prediction fails 164/164. MBPP-C references pass on 952 of the 973 tasks that compile; we keep those 952.
- Our first harness took the C and asserts from LLM4Decompile instead. Six of its HumanEval problems (0, 1, 101, 125, 137, 162) do not match GG's rows, and the same predictions scored 68/164 instead of 72. We now use only the authors' tests.

**Contamination (checked against GG's public SFT data).**

- HumanEval-C: no copies. Only 7 one-line functions compile to assembly that also appears in the SFT data.
- BringUp: 12 of 65 programs are copies of SFT files. The paper counts 3 of them as passes. We report BringUp on all 65 and on the clean 53.
- MBPP-C vs HumanEval-C: 13 MBPP tasks are the same problems as 12 HumanEval problems; 9 are in train and 4 in dev. The HumanEval gain holds without them (see GRPO (TRL default) results).
- Qwen2.5-Coder's own pre-training data cannot be checked: its decontamination matched Python 10-grams, which misses C.

**Engineering findings.**

- QLoRA was dropped. TRL merges the LoRA update into the 4-bit weights before sending them to vLLM, and the 4-bit rounding erases most of the update. Generating without vLLM was 6–9x slower. Full-parameter training matches how GG did its own SFT.
- transformers 4.x silently loads checkpoints saved by transformers 5.x with the wrong RoPE base (10,000 instead of 1,000,000). Logits differ by up to 16.5 and the top token agrees at only 1,626 of 1,802 positions. Our eval code now checks the RoPE base at load time.
- Across software stacks, all 157 compared HumanEval outputs have the same tokens, but per-token log-probabilities differ by up to 0.28–0.37. That is enough to change beam-search results by a few problems.
- Our two RTX 4070 Ti SUPER machines produce identical outputs, so either can run HumanEval and dev.

## Baseline arms (planned)

Readers take plain "GRPO" to mean DeepSeekMath's algorithm, and our run differs from it. We therefore add two named algorithms as baselines. All arms share one setup; only the switches below differ, and each switch comes from a named paper.

| Switch | GRPO (DeepSeekMath) | GRPO (TRL default), our runs | Dr. GRPO |
| --- | --- | --- | --- |
| Loss aggregation | Mean over each completion's tokens, then over completions | Sum over all tokens in the batch ÷ number of tokens (DAPO's token-level loss) | Sum over all tokens ÷ a constant (8 × 4,096) |
| Advantage | (reward − group mean) / group std | (reward − group mean) / group std | reward − group mean |
| KL penalty | beta 0.04 | none | none |
| Truncated completions | In the loss | Excluded (DAPO's overlong filtering) | In the loss |
| TRL flags on top of `grpo.yaml` | `loss_type grpo`, `beta 0.04`, `mask_truncated_completions false` | none | `loss_type dr_grpo`, `scale_rewards none`, `mask_truncated_completions false` |

- Shared by all arms: model, data, 8 completions per prompt, 128 completions per step, 106 steps, lr 1e-6 with linear decay, 4,096-token cap, temperature 1, one optimizer step per rollout batch, and the same eval protocol and hardware.
- Concessions against the source papers, stated once for all arms: DeepSeekMath used 64 completions per prompt, 1,024 prompts per step, a 1,024-token cap and a learned reward model; Dr. GRPO used a constant LR and a 3,000-token cap. Dr. GRPO's own small-model setup (8 completions per prompt, lr 1e-6, temperature 1, one update per rollout batch) is otherwise close to ours.
- DAPO is deferred: TRL 1.16 has no dynamic sampling, and DAPO's wider clip does nothing with one optimizer step per rollout batch.
- Seed 0 of each new arm comes first; 3 seeds per arm later.

## Pending runs and open questions

The baseline arms take the training GPU first, so seed 1 of GRPO (TRL default) pauses at checkpoint 27 and resumes after them. A resumed run is not bit-identical to an uninterrupted one (vLLM's sampler restarts), so we will report it as resumed at step 27. Until seed 1 finishes on Oct 24, the HumanEval-C gain rests on a single training run.

| Run | Gives | Expected (GST) |
| --- | --- | --- |
| BringUp, GRPO (TRL default) seed 0 checkpoint 106 | Trained-model BringUp, vs base 6/65 (beam 8) and 5/65 (greedy) | Beam 8 generated, scoring pending; greedy about 12:00–13:00, Oct 23 |
| Dev beam 8, GRPO (TRL default) seed 1 checkpoint 27 | First point of the second dev curve | Midday, Oct 23 |
| GRPO (DeepSeekMath) training, seed 0 | First named baseline | About 7–7.5 h after launch: about 18:30–19:00 if it starts by 11:30, Oct 23 |
| Dev beam 8 and sampled, GRPO (DeepSeekMath) checkpoints 27/54/81/106 | Its dev learning curve | As each checkpoint lands |
| HumanEval-C beam 8, GRPO (DeepSeekMath) checkpoint 106 | Baseline against GRPO (TRL default)'s 87/164 | About 20:15 (split over two GPUs) to 22:00, Oct 23 |
| Dr. GRPO training, seed 0 | Second named baseline | About 01:00–02:00, Oct 24 |
| GRPO (TRL default) seed 1, resumed from checkpoint 27 | Second seed: does the +15 on HumanEval-C replicate? | Training about 07:00, Oct 24; evals after |
| BringUp for the new arms and seed 1 | BringUp for every arm | After the matching training run, Oct 24 |

**Open questions.**

1. Why does HumanEval-C gain 15 problems under beam 8 while dev stays flat?
2. Why does base BringUp score 6/65 here against the paper's 10/65?
3. The paper says sampling was disabled, but its released script samples during beam search (temperature 0.7, inherited from the model config). Which setting produced the paper's 74/164?
4. 57% of prompt groups give no gradient. Should the next run use dynamic sampling or a harder prompt mix?
5. Next training data, if needed: sources with real tests are Codeforces (890 problems with complete official tests), AOJ (up to 1,832) and AtCoder (1,391; license unclear). Every other source we found has only synthetic tests or none.
