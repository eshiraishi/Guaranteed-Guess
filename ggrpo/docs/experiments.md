# ggrpo experiment design: which GRPO/RLVR choices help on GG x86 → ARMv8 O2

**Status (Oct 23, 2026): designed, not started.** No GPU is available right now. Nothing here is implemented.
When resources free up, start at Phase 0.

The goal is to learn which RLVR choices (reward shaping, GRPO/DAPO-family losses, data and sampling) improve
`gg-armv8-O2` (72/164 on HumanEval-C O2, beam 8), with each comparison changing one thing. A value with no source in
the paper, the data or a cited study stays `TODO`, as in `grpo.yaml`.

## Decisions
- **Order (Oct 23).** Finish one run, fix the training regime (learning rate and length), measure the run-to-run
  noise, then test only the variants that the run's diagnostics call for, plus the partial-credit reward.
  - The first draft screened about 9 variants up front. The evidence below says variant effects are small and depend
    on the setting.
  - Our regime (LR 1e-6, 106 steps) may be too small for any variant to show an effect.
- **Control (Oct 23): TRL-default GRPO.** This is the current `grpo.yaml`: binary reward, `loss_type: dapo`,
  `scale_rewards: group`, β 0, ε 0.2, μ 1, at the LR and length fixed in Phase 2.
  - It is the recipe most people run today: a binary verifiable reward (DeepSeek-R1, DeepCoder), token-level
    aggregation and β 0 (DAPO, Dr. GRPO), and group std (GRPO).
  - It is a mix, not one paper's algorithm. If the paper needs the canonical algorithm, the original GRPO
    (DeepSeekMath 2402.03300: `loss_type grpo`, β 0.04) can be added as an arm.
- **Seeds (Oct 23).** 3 seeds for the control only. Screen variants with 1 seed (seed 0, the same data order as
  control seed 0), and add seeds only for results that go into the paper. RLVR method papers usually report one run
  per configuration (DAPO, Dr. GRPO, DeepCoder). Hochlehnert et al. show that this hides large seed variance on small
  benchmarks, and ours are small.

## Verified facts that shape the design (Oct 23)
- **Training is exactly on-policy today.** `gradient_accumulation_steps` = `steps_per_generation` = 32 and
  `num_iterations` = 1, so every rollout batch gets exactly one optimizer step.
  - The old log-probs are computed before that step, so the PPO ratio is 1 (TRL 1.16 `grpo_trainer.py` ~L2642 and
    L3066).
  - The clipping variants therefore do nothing until rollouts are reused (`num_iterations` > 1): `epsilon_high` (DAPO
    clip-higher), `delta` (INTELLECT-2), `cispo`, `sapo`, and GSPO's `importance_sampling_level`.
  - On-policy, the loss family reduces to two choices: token aggregation (`loss_type`) and advantage scaling
    (`scale_rewards`).
- **RLOO needs no separate trainer.** Its leave-one-out advantage is G/(G−1) × (r − group mean), which is
  `scale_rewards: none` up to a constant, and Adam is nearly invariant to a constant gradient scale.
- **TRL 1.16 covers every arm below through config,** except the new reward functions. The relevant options are:
  - `loss_type` (grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo, vespo) and `scale_rewards` (group, batch, none);
  - `epsilon_high`, `delta`, `importance_sampling_level`, `num_iterations`, `steps_per_generation`;
  - `top_entropy_quantile`, `entropy_coef`, `use_adaptive_entropy`, `off_policy_mask_threshold`;
  - `reward_weights` and `multi_objective_aggregation` (including GDPO's `normalize_then_sum`);
  - `trl.rewards.get_soft_overlong_punishment`.

  TRL has no DAPO-style dynamic resampling, so filtering has to be done offline. It logs `frac_reward_zero_std`.
- **Partial credit is coarse.** Assert counts per driver (`assert(` occurrences):

  | Split | Tasks | Median asserts | Max | Drivers with no assert |
  |---|---|---|---|---|
  | train | 858 | 3 | 20 | 3 |
  | dev | 94 | 3 | 20 | 0 |
  | eval | 164 | 6 | 26 | 2 |

  153 of 858 train drivers contain a loop, and at least 90 assert inside one. Asserts must therefore be counted at run
  time, not in the source text. A failed assert ends in SIGABRT, which `harness.py` reports as `crash`, the same
  verdict as a segfault.
- **Gap to fix first:** `server.py` loads train + eval tasks only, so scoring `data/dev.jsonl` returns 404.

## Constraints from other Oct 23 findings
- **Eval lanes.** The Blackwell (totoro) only trains. Census, dev and HumanEval scoring run on the RTX 4070 Ti SUPERs
  (totoro GPU1 or ponyo; modern stack; identical outputs). BringUp runs on the student-lab RTX 5000 Ada (transformers
  4.50, at most 2 jobs).
  - Every base-vs-trained comparison uses one GPU type and one stack.
  - Checkpoints saved by transformers 5.x lack a top-level `rope_theta`, and 4.50 would silently use 10000, so BringUp
    evals of trained checkpoints need a fixed config.
  - The lane scripts are in `ops/lanes/`.
- **Contamination.**
  - 13 MBPP tasks are same-problem twins of 12 HumanEval problems, and 9 of them are in train: mbpp/686, 812, 556,
    499, 604, 872, 67, 1, 31, 32, 98, 754, 957.
  - RL on them inflates HumanEval. Decide the agreed decontamination toggle before Phase 1, because it changes the
    train set (and the step count) of every run.
  - Report HumanEval on all 164 and on the problems without a twin.
  - 12 of 65 BringUp programs are copies of GG SFT sources, so report BringUp on 65 and on the clean 53.

## Evidence from the literature (and how strong it is)
- **Variant effects are small and depend on the setting.**
  - ScaleRL (2510.13786): loss aggregation, normalization, curriculum and the off-policy algorithm mostly change
    compute efficiency, not where training plateaus. Loss type and fp32 logits do move the plateau.
  - Tricks or Traps (2508.08221): most tricks help or hurt depending on model, size and reward scale, and a two-trick
    recipe (group-mean/batch-std advantages and token-level aggregation) matched the complex ones.
  - Both studies are on math tasks with larger models, so whether they transfer here is an assumption.
- **Implementation details and hyperparameters can matter as much as the algorithm.**
  - Engstrom et al. (2005.12729): code-level details explain most of PPO's gain over TRPO. That is classic deep RL,
    not LLMs.
  - For LLM GRPO, controlled hyperparameter sweeps are thin. Learning rate was swept only for LoRA (2512.23165:
    1e-6 to 1e-5).
  - The group-size evidence conflicts. At a fixed batch, G from 2 to 16 performs about the same ("It Takes Two: Your
    GRPO Is Secretly DPO"), while 2509.25300 finds the best G shifts with the training budget.
- **Many claimed RL gains shrink under careful evaluation.** Hochlehnert et al. (2504.07086) find that seed and
  decoding variance is comparable to the gains papers claim, on small benchmarks.
- **Partial credit has mixed evidence.**
  - The pass-rate reward study (2605.02944): helps early, slightly better pass@1 at convergence, worse pass@k at
    large k, and nearly the same problems solved as with the binary reward.
  - VeRPO (2601.03525): plain test fractions invite overfitting to easy tests. Its difficulty-weighted tests report
    +8.8 pass@1 (the authors' own claim).
  - DeepCoder: avoided K/N credit because of hacking.
  - CodeRL (2207.01780): ladder rewards of −1 compile error, −0.6 runtime error, −0.3 failed test, +1 pass.
  - Signal Reshaping (2605.07276): compile-only rewards favour syntactic shortcuts.
- **Nothing shows that a specific variant matters for assembly transpilation.** The closest domain, SuperCoder
  (2505.11480), used plain PPO/GRPO with a test reward. GG and its precursor CRT used no RL.
- **The risk for our regime:** LR 1e-6 for 106 steps is a very small budget. Adam moves each weight by about one LR
  per step, so at most about 1e-4 in total. Recipes for 1.5B models that report gains ran on the order of 1–2k steps.
  - If the control's curves are flat, every variant ties.
  - If it saturates early, only efficiency differs.

## Reference points
| Name | What it is | Question it answers |
|---|---|---|
| base | `gg-armv8-O2`, no training | Does RL help at all? |
| RFT | Rejection-sampling SFT (STaR/RAFT-style) on the base model's passing samples, `trl sft` | Does RL beat simple self-distillation? |
| control | TRL-default GRPO (see Decisions) | Does variant X help? |

## Plan
### Phase 0: diagnostics (no training; minutes to about 1 h each, on the 4070 lane)
0. **Decontamination toggle.** Decide whether to drop the 9 HumanEval twins from train, rebuild `data/train.jsonl`,
   and recompute `max_steps`.
1. **Base rollout census.** vLLM at the training settings (T 1.0, top_k 0, top_p 1, rep 1.0, 4096 tokens), n = 16
   per prompt, on train and dev, scored with verdicts. Report:
   - per-prompt pass rates and the share of prompts at 0/16 and 16/16;
   - the expected `frac_reward_zero_std` at G = 8 and G = 16 (by subsampling);
   - the verdict mix among failures;
   - the truncation rate.

   The same samples are the RFT data.
2. **Eval noise.** Bootstrap SE of avg@16 on dev and HumanEval for the base model, and how avg@16 relates to beam-8
   72/164. The SE is the smallest effect we can detect.

### Phase 1: the first control run
- Run `grpo.yaml` as is, with seed 0. Score dev avg@16 at checkpoints 27, 54, 81 and 106.
- Read reward, `frac_reward_zero_std`, entropy, completion length, truncation and `verdict/*`.
- **Decision:**
  - If dev avg@16 rises by more than 2 eval SEs, the run is in a working regime.
  - If it is flat, the LR or the length is too small.
  - If entropy or reward collapses, the LR is too high.

### Phase 2: the training regime
- **LR:** 1e-6 (Phase 1), 3e-6 and 1e-5, 1 seed, 106 steps. `TODO`: this is a log-spaced proposal over the range
  2512.23165 swept for LoRA; TRL's default is 1e-6. Pick the highest LR with no collapse whose dev curve rises fastest.
- **Length:** if dev is still rising at step 106, extend the run (more epochs over the 858 prompts) and fix the step
  count from the curve.
- **RFT:** run it here too. Its LR and epochs are `TODO`; take them from GG's SFT recipe if the paper or repo states
  them.
- The result is the control regime. Nothing in Phase 4 runs before it is fixed.

### Phase 3: noise floor
- Run the control at the fixed regime with seeds 1 and 2. Every variant result is variant − control at seed 0, so
  the control's spread is the noise floor of that difference.

### Phase 4: targeted variants (1 seed each)
| Arm | Change | Run it when |
|---|---|---|
| **R1 assert fraction** | `reward_fraction`: asserts passed / asserts the reference executes; 1.0 only on pass | Always: this is the task's research question |
| D1 filter | `data.train_ids` = prompts the base solves 1–15 of 16 times; same step count | `frac_reward_zero_std` ≥ 25% (`TODO` threshold) |
| D2 G = 16 | `--num_generations 16`, 8 prompts per step (the same 128 completions) | Same trigger, if the census predicts a large drop at G = 16 |
| N2 Dr. GRPO | `--loss_type dr_grpo --scale_rewards none` (≈ RLOO) | Many prompts have extreme pass rates, so std scaling gives them large weights |
| O1 / O2 reuse | `--num_iterations 2` (O2 adds `--epsilon_high 0.28`) | The reward server limits throughput, or curves rise slowly per rollout. Compare at the same number of rollout batches. |
| L1 overlong | soft overlong punishment instead of `mask_truncated_completions` | Truncation ≥ 5% (`TODO` threshold) |
| R2 ladder | CodeRL values over the verdicts (needs assert-abort split from other crashes) | Build failures are a large share of failures |

Before R1:
- **Stub audit.** Score `mov w0,#0; ret`, `mov w0,#1; ret` and bare `ret` under R1. If stubs earn real credit, R1
  also logs a stub rate, and VeRPO-style difficulty weighting is the follow-up.

**Promotion rule.** Dev avg@16 at the final checkpoint beats the control's seed mean by more than the control's seed
range, with no hacking signal (stub rate, verdict shift). Promoted arms get seeds 1 and 2.

**Not planned, unless a diagnostic calls for it:**
- KL on its own: DAPO and Dr. GRPO drop it.
- `top_entropy_quantile` 0.2: the 80/20 paper's gains grow with model size.
- Pass@k objectives: the headline metric is beam 8.
- CISPO / GSPO / SAPO: they only matter after O1 shows that reuse helps.

### Phase 5: confirmation and reporting
- Combine the promoted arms (3 seeds).
- **Report every arm:**
  - dev curves;
  - on held-out HumanEval (all 164 and the problems without an MBPP twin): avg@16, pass@k for k = 1..16 (unbiased
    estimator; shows sharpening vs coverage, per Yue et al. 2504.13837), and beam-8 pass@1;
  - BringUp O2 (beam 8; 65 and clean 53) for base, control and the best recipe.
- **Statistics:** paired per-task bootstrap against the control (with seeds: resample seeds and tasks). Arms are chosen
  on dev only, and the reported model is the final checkpoint (`docs/code_plan.md`).

## Evaluation protocol
- **Selection** uses MBPP-C dev only. HumanEval-C and BringUp stay held out.
- **Dev:** avg@16 at the training sampling settings at every checkpoint, and beam 8 at the final checkpoint.
- **Training diagnostics per run:** reward mean/std, `frac_reward_zero_std`, entropy, completion length, truncation,
  `verdict/*`, and the binary pass rate for every reward function (so curves are comparable across rewards).
  Off-policy arms also log the clip ratios.
- **Equal compute:** compare at the same number of rollout batches, and report wall-clock and reward-server seconds.

## Code needed (none written yet; show samples before implementing)
1. `src/ggrpo/server.py`: load dev tasks.
2. `src/ggrpo/generate.py`: an n-sample vLLM path, for the census and avg@16. HF beam search stays for the paper
   metric.
3. `src/ggrpo/score.py`: multi-sample scoring (per-task rate, avg@n, pass@k) and a paired comparison against a baseline
   file.
4. For R1 only:
   - `src/ggrpo/harness.py`: a shim `assert.h`, found first through `-I`, that counts passed and failed asserts with
     unbuffered writes so the counts survive a later crash. It is used for the training reward on failures only; the
     binary verdict and the eval keep the unmodified driver.
   - `src/ggrpo/build.py`: `asserts_total` per task, from the reference ARM.
   - `src/ggrpo/reward.py`: `reward_fraction`.
   - `config.toml [reward]`.
5. Filtering reuses `data.train_ids`. RFT uses `trl sft` with its own YAML, so there is no train script.

## Budget estimate (Blackwell training time only; 106 steps ≈ 8–10 h at 4–6 min/step; evals run on the other lanes)
| Phase | Runs | GPU time |
|---|---|---|
| Phase 0 | none | about 2 h |
| Phase 1 | 1 | about 0.4 day |
| Phase 2 | 2 LR runs (+ extension) + RFT | about 1–1.5 days |
| Phase 3 | 2 | about 1 day |
| Phase 4 | R1 + 2–4 triggered arms | about 1.5–2 days |
| Phase 5 | seeds for finalists + combination + evals | about 2–3 days |
| **Total** | | **about 6–8 GPU-days, with checkpoints to stop early** |

## Verification
- **Census sanity:** the reference ARM passes every task, and base dev avg@16 is consistent with the base beam-8 level.
- **Server:** dev scoring returns verdicts, not 404.
- **Multi-sample scoring:** run `ggrpo.score` end to end on the base model and the Phase 1 checkpoints.
- **R1 shim unit tests** (VM): correct counts with loop asserts and a mid-run crash. Binary verdicts on the 164 beam-8
  predictions match the current harness exactly.
- **Smoke test for each new arm** (10 steps): it starts and logs `pass` and `verdict/*`. O1 shows nonzero clip ratios.

## Sources
- ScaleRL, The Art of Scaling RL Compute for LLMs: https://arxiv.org/abs/2510.13786
- Tricks or Traps? A Deep Dive into RL for LLM Reasoning: https://arxiv.org/abs/2508.08221
- Engstrom et al., Implementation Matters in Deep Policy Gradients: https://arxiv.org/abs/2005.12729
- Hochlehnert et al., A Sober Look at Progress in LM Reasoning: https://arxiv.org/abs/2504.07086
- Evaluating Parameter Efficient Methods for RLVR: https://arxiv.org/abs/2512.23165
- Scaling Behaviors of LLM RL Post-Training: https://arxiv.org/abs/2509.25300
- It Takes Two: Your GRPO Is Secretly DPO: https://liner.com/review/it-takes-two-your-grpo-is-secretly-dpo
- Exploring Pass-Rate Reward in RL for Code Generation: https://arxiv.org/abs/2605.02944
- VeRPO / Beyond Binary: https://arxiv.org/abs/2601.03525
- DeepCoder: https://together.ai/blog/deepcoder
- CodeRL: https://arxiv.org/abs/2207.01780
- Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair: https://arxiv.org/abs/2605.07276
- SuperCoder: https://arxiv.org/abs/2505.11480
- Posterior-GRPO: https://arxiv.org/abs/2508.05170
- Dr. GRPO, Understanding R1-Zero-Like Training: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Yue et al., Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?: https://arxiv.org/abs/2504.13837
- Guaranteed Guess: https://arxiv.org/abs/2506.14606
