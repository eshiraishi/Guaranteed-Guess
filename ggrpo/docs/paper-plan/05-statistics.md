# 05. Statistical evaluation protocol

**Status (Oct 24, 2026): proposed protocol, to be pre-registered before the confirmatory runs.**
Tags: [V] verified, [R] repo, [L] unverified literature or standard statistics, [P] proposal.

## Summary
- **Our current benchmarks can detect only large effects.** These are paired tests at α = 0.05 with 80% power, using
  Miller's formula and our measured discordance:

  | Set | Tasks | Detectable effect |
  |---|---|---|
  | HumanEval-C | 164 | ≈ 7 pp |
  | MBPP-C dev | 94 | ≈ 9 pp |
  | clean BringUp | 53 | ≈ 12 pp |

  The flat dev result is therefore **inconclusive**, not a null [V+R]. The new **GG-XL** held-out set of at least 1,000
  problems brings the detectable effect to about 3 pp [P] ([01](01-data.md)).
- **Paired, per-task analysis everywhere** [V]:
  - beam-8 (one binary outcome per task): exact McNemar, which is the same as the exact sign test;
  - sampled metrics: average per task, then a paired permutation test or bootstrap clustered on the task (or problem);
  - several training seeds: average per task over seeds before the paired test.
- **Small-N-valid intervals:** Wilson or Beta(1+s, 1+f) per arm, and a paired Bayes or exact interval for differences.
  Never report mean ± 1.96·SE: CLT intervals under-cover below a few hundred items [V].
- **Holm** within pre-declared comparison families. Six uncorrected pairwise tests among 4 models give a family-wise
  error of about 0.19 [V].
- **The training-seed variance of RLVR at our scale is unknown** [V gap], so we measure it before fixing the seed count.
- **Endpoints (planner's decision, Oct 24).** Co-primary: **beam-8 pass@1** (GG's protocol and the deployment decoding)
  and **avg@16 sampled pass@1** (the distribution RL optimizes), both on GG-XL. Everything else is secondary. Reasons are
  in §1.

## 1. Endpoints
| Endpoint | Definition | Why | Role |
|---|---|---|---|
| **E1: beam-8 pass@1** | beam 8, `do_sample=False`, repetition penalty 1.1, cap 32k − prompt; all tests must pass | GG's reported protocol [R]; nearly deterministic, so no resampling is needed; the decoding a user would deploy | co-primary |
| **E2: avg@16** | mean over 16 samples per task at the training sampling settings (T 1.0, top-p 1, top-k 0, rep 1.0), averaged per task first | RL optimizes this distribution. Run 1 moved it on dev (+3 pp, sign test p = 0.016) while dev beam-8 stayed flat [R] | co-primary |
| E3: pass@k, k = 1…16 | unbiased estimator from the same 16 samples (Chen et al. 2021) [L] | separates sharpening from new coverage; run 1's pass@8 was flat [R] | secondary |
| E4: test-verified best-of-n | a task counts as solved if any of n candidates passes its tests (= pass@n) | GG's framing: tests make a correct guess verifiable, so a user with tests can keep the first passing candidate | secondary; report n and its cost |
| E5: HumanEval-C beam-8 | E1 on 164 and on the 152 non-twin problems | direct comparison with GG's 74/164 [R] | secondary, reported |
| E6: BringUp beam-8 | E1 on all 65 and the clean 53 | whole-program generalization | secondary (detectable effect ≈ 12 pp) |
| E7: efficiency | geometric-mean speedup and size against the clang reference, on passing outputs only | GG's systems claim | secondary ([02](02-harness.md) §6) |

**Per target:** every endpoint is reported per target (ARMv8-macOS, AArch64-Linux, RISC-V64) and per optimization level.
Targets are separate families (§4).

**Two co-primaries, not one.** E1 alone hides gains that only show in sampling. E2 alone breaks comparability with GG.
Holm over the two keeps the error rate at α [P].

## 2. Tests and intervals
### 2.1 One binary outcome per task (E1, E5, E6)
- **Per arm:** accuracy with a **Wilson** interval, (p̂ + z²/2n ± z·sqrt(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), or a
  **Beta(1+s, 1+f)** credible interval [V]. Clopper–Pearson is valid but too wide [V].
- **Per comparison:** report the discordant counts b (up) and c (down), Δ = (b − c)/n, and the **exact two-sided McNemar**
  p = 2·Σ_{i ≤ min(b,c)} C(b+c, i)/2^{b+c} [V]. Example: run 1 on HumanEval, 16 up / 1 down, gives p = 0.000275 [V+R].
- **Interval for Δ:** the paired Bayes method of Bowyer et al. (Beta(1,1) marginals with a Gaussian copula,
  importance-sampled) [V], or an exact or score interval for paired proportions [L]. At n ≈ 164 the gain over the
  paired CLT interval is small (coverage 0.937 vs nominal 0.95 for Wald at p ≈ 0.87) [V], but the method is standard and cheap.

### 2.2 Several samples per task (E2, E3)
- Average the K samples per task first, then do the inference over tasks. This is equivalent to clustering on the task.
  Clustered SEs ran 1.1–3.05x the naive ones in Miller's examples [V].
- **Test:** a paired sign-flip permutation test over tasks (exact when n ≤ 12, Monte Carlo otherwise), or a paired
  bootstrap over tasks [V]. Report the per-task correlation between the two arms [V].
- **GG-XL has several solutions per problem**, so cluster on the **problem**, not the prompt.
- **K = 16:** it captures about 94% of the variance reduction resampling can give, K = 8 about 87.5% and K = 32 about
  97% [V, (1+2/K)/3 rule]. Hochlehnert et al. recommend 30 or more eval seeds on 30-problem benchmarks [V]. Ours are
  larger, so 16 is the default, and 32 when cheap.

### 2.3 Several training seeds
- Average each task over seeds (and over samples), run the paired test against the reference arm's seed-averaged
  per-task scores, and report every seed's number and the seed SD [V, Hochlehnert-style].
- **Variance decomposition [P]:** fit score_{t,s,k} = μ + task_t + seed_s + (task×seed)_{t,s} + ε to the noise-floor
  experiment (§5). The variance components then set the seed count and K for the confirmatory runs.
  - One proposed linear-mixed-model equivalence was refuted 1–2 [V], so use the model descriptively and keep the
    paired tests above as the confirmatory analysis.

### 2.4 Curves and scaling fits
- Fit ScaleRL's sigmoid per arm, R − R0 = (A − R0)/(1 + (C_mid/C)^B), on dev after at least one epoch. ScaleRL's stated
  seed band is ±0.02 on A (±0.015 over 3 seeds) [V].
- Compare the fitted A values with a bootstrap over seeds. A difference below 0.02 is noise [V]. Do not rank arms on early
  reward, because early rankings invert [V].

### 2.5 Claims of "no effect"
- A null needs an **equivalence test** (TOST) with a margin fixed in advance, Δ_eq = 3 pp on GG-XL [L standard; margin P].
- Without it, a null is "inconclusive below the detectable effect". The margin choice itself was not verified [V gap].

## 3. Power and sample size
- **Detectable effect:** δ = (z_{α/2} + z_β)·sqrt((ω² + σ_A²/K_A + σ_B²/K_B)/n), with z_{0.025} + z_{0.20} = 2.80 [V].
  - ω² is the variance of the paired per-task difference. For a single binary outcome it is about the discordance rate:
    17/164 → ω² ≈ 0.095 on HumanEval in run 1 [V+R].
  - σ²/K is the within-task sampling variance over K samples.
- **Detectable effect by eval set, for E1:**

  | Set | n | Detectable effect | Use |
  |---|---|---|---|
  | HumanEval-C | 164 | ≈ 6.8 pp | comparability |
  | MBPP-C dev | 94 | ≈ 8.9 pp | selection |
  | BringUp all / clean | 65 / 53 | ≈ 10.7 / 11.9 pp | supporting |
  | **GG-XL** | 1,000 / 1,300 | ≈ 2.7 / 2.4 pp | **primary** |

- **Recompute everything with measured ω² and σ²** from the noise-floor experiment before freezing the plan. These
  numbers assume HumanEval's discordance carries over [V caveat].
- **Do not subsample with IRT** (tinyBenchmarks-style) to save compute. It raised seed SD 2–10x and flipped about 12% of
  pairwise orderings [V].

## 4. Multiplicity
Each family is pre-declared, and Holm is applied within each family, at α = 0.05 per family [V+P].

| Family | Comparisons | Endpoints |
|---|---|---|
| **P1-main** | final recipe vs GG base; vs SFT data-control; vs RFT; vs GRPO (TRL default) | E1, E2 on GG-XL |
| P1-ablations | each ablation vs its parent arm (one switch each) | E2 on GG-XL |
| P1-targets | final recipe vs its SFT start, per target × optimization level | E1 per target |
| P2-main | R+S vs R-only; S-only vs base | E2 on tier-R evals |
| P2-validity | stated as estimates with CIs (FPR, FNR), not tests | — |

- Report raw and Holm-adjusted p-values side by side.
- Exploratory analyses are labeled as such and are not corrected.

## 5. Noise floor experiment (runs before the confirmatory phase)
- **Design [P]:**
  - Run the RL control recipe with **3 seeds** at the fixed regime, with a different data order per seed.
  - Score each seed on dev and GG-XL with E1 and E2 (K = 16), plus base GG with K = 32 for the sampling variance.
- **Outputs:**
  - the seed SD;
  - ω² and σ² per set;
  - variance components (§2.3);
  - the detectable effects recomputed.
- **Decision rule:**
  - If the seed SD on E2 is below a third of the GG-XL detectable effect, use **3 seeds** per finalist arm.
  - Otherwise use **5**. The headline recipe always gets 5.
  - The only verified seed evidence is pretraining at 7B, with HumanEval seed SD 1.11 pp against a test CI of 3.98 pp [V].
    RLVR may differ.

## 6. Selection, fairness and compute matching
- **Selection on dev only.** GG-XL, HumanEval-C and BringUp are never used to pick arms, checkpoints or hyperparameters
  [R rule].
- **Checkpoints:** the reported model is the **final** checkpoint of a pre-declared length. Dev may be used only to choose
  a regime (LR, length) before the confirmatory runs, never per arm afterwards [R, code_plan].
- **Equal tuning budget:** every baseline that has hyperparameters (SFT data-control, RFT, each GRPO variant) gets the same
  LR grid and the same number of tuning runs as the proposed recipe [P]. Reviewers ask for this. This topic had no
  verified claim [V gap], so this is standard practice, not a cited result.
- **Compute matching:** compare arms at equal **generated tokens** and **GPU-hours**, not steps, whenever an arm changes
  per-step cost (curriculum, filtering, G, reuse). Report reward CPU-seconds too ([02](02-harness.md)).

## 7. Pre-registration
Before Phase E ([06](06-schedule.md)), commit `docs/paper-plan/prereg.md` with:
- the endpoints;
- the families and the Holm procedure;
- the eval sets with their content hashes;
- the seed counts;
- the recomputed detectable effects;
- the equivalence margin;
- the checkpoint rule;
- the exclusion lists;
- the exact decoding settings.

Optionally deposit it on OSF. Any deviation is reported in the paper.

## 8. Reporting checklist
- The table of every arm that was run, including failed and abandoned ones, with configs and seeds.
- Per arm, per endpoint:
  - accuracy with its Wilson/Beta interval;
  - Δ vs the reference arm with b/c counts;
  - raw and Holm p-values;
  - the per-task correlation.
- The detectable effect for each eval set, next to every null claim.
- Clean-subset results: HumanEval-C 152 and BringUp 53, plus the contamination audit method ([01](01-data.md) §6).
- Compute: GPU-hours, generated tokens, reward CPU-seconds and wall-clock.
- Released per-task score files, so others can rerun every test.

## 9. Gaps (no verified evidence in this run)
- Equal-budget tuning norms and reviewer expectations for RLVR papers.
- Time-split evaluation.
- rliable/IQM, Almost Stochastic Order, Dodge et al.'s expected max performance, TOST margins, and the variance of the
  pass@k estimator.

These are standard tools, used here on judgment [L]. Verify them before citing.

## Sources
Verified claims: [evidence/05-stats.md](evidence/05-stats.md).
- Miller, Adding Error Bars to Evals: arXiv 2411.00640; Anthropic, A statistical approach to model evaluations
- Bowyer et al., Don't rely on the CLT (ICML 2025): https://proceedings.mlr.press/v267/bowyer25a.html
- evalci: arXiv 2607.04429; multiple comparisons in LLM evals: arXiv 2608.22659
- Hochlehnert et al., A Sober Look at Progress in LM Reasoning: arXiv 2504.07086
- Madaan et al., Quantifying variance in evaluation benchmarks: arXiv 2406.10229
- Hariri et al., Don't Pass@k (ICLR 2026): arXiv 2510.04265; ScaleRL: arXiv 2510.13786
