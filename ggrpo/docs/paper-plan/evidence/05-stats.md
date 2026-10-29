# Evidence: 05-stats (deep-research output, Oct 24, 2026)

Raw synthesis of the 3-vote verified claims. Fields: confidence | claim, SRC (sources), EV (evidence and vote). Sections after the findings: caveats, open questions, refuted claims, sources, stats.

SUMMARY: Our benchmarks are small: HumanEval-C (164 tasks), the MBPP-C dev set (94) and the 53 clean BringUp programs. On those, a paired, deterministic beam-8 comparison can detect only large effects. At alpha=0.05 and 80% power, Miller's (2024) Eq. 10 with our measured discordant rate gives a minimum detectable effect (MDE) of about 7, 9 and 12 pp. Sampling each task K times for avg@n removes only the within-task noise, so even at K=16 the MDE for sampled pass@1 stays at 7-8 pp or more. The verified practice is a paired per-task analysis. For beam-8, use the exact McNemar test, which is the same as an exact sign test on the discordant tasks; our 16 up / 1 down gives two-sided p=0.000275. For avg@n, average the samples per task first, then run a paired test over tasks with the task as the cluster. Across the pre-declared set of arm comparisons, apply Holm, because uncorrected pairwise tests among 4 models give a family-wise error rate of about 0.20. For intervals, use Wilson or a Beta(1+s,1+f) posterior for one arm's accuracy and a paired Bayesian (or exact) interval for differences, not mean±1.96·SE, which under-covers below a few hundred items (only modestly at N=164, p≈0.87). The verified evidence on training-seed variance comes only from 7B pretraining (HumanEval seed SD 1.11 pp vs a 3.98 pp CI half-width), and the evidence on eval seeds only from math benchmarks (≥30 recommended, 10 used). We therefore have to measure RLVR run-to-run variance at our scale ourselves. Checkpoint selection, fairness of tuning budgets, contamination methods and what A* reviewers ask for were not verified in this run.

- high | Power analysis: a 164-task benchmark can detect only large effects, so run the analysis before evaluating and report the MDE per benchmark. Sample size: n = (z_{a/2}+z_b)^2 (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2. MDE: delta = (z_{a/2}+z_b) * sqrt((omega^2 + sigma_A^2/K_A + sigma_B^2/K_B)/n), with z_{0.025}+z_{0.20} = 2.80. Here omega^2 is the variance of the paired per-task difference in conditional mean scores; for one binary outcome per task it is about the discordant rate. sigma^2/K is the within-task sampling variance divided by the K samples per task. With omega^2=1/9 and no sampling noise, detecting 3 pp takes about 969 tasks, hence Miller's advice that new evals have at least 1,000 questions. Applied to GG (my own arithmetic; the paper does not mention 164-task benchmarks): HumanEval-C beam-8 with omega^2≈0.095, from 17/164 discordant, has an MDE of about 6.8 pp. Assuming a similar omega^2, the 94-task dev set has about 8.9 pp, all 65 BringUp programs about 10.7 pp, and the 53 clean ones about 11.9 pp. A flat dev result is therefore inconclusive below roughly 9 pp. It is not evidence of no effect.
  SRC: https://arxiv.org/abs/2411.00640 https://anthropic.com/research/statistical-approach-to-model-evals
  EV: Verifiers checked the paper's Eq. 9-10 and its worked example ((1.96+0.8416)^2 × 0.111/0.0009 ≈ 969) in the full text. The Anthropic post separately recommends an a-priori power analysis and says small differences will likely go undetected on small evals. Both are standard power analysis. Qualifications: the ω^2=1/9 example parameters are fictional, the source has an internal arithmetic slip (correlation 0.5 with Var(x)=1/12 gives ω^2=1/12, not 1/9), and the normal-approximation formula is less reliable at n<few hundred (Bowyer 2025). The post and the paper are companion works by the same author, so this is effectively one primary line of evidence, backed by textbook statistics.

- high | For sampled metrics (avg@n, pass@k), resampling has diminishing returns, and the K samples per task must be clustered on the task. Var(mu_hat) = (Var(x) + E[sigma_i^2]/K)/n. Resampling shrinks only the within-task term. The between-task term Var(x) is a property of the task population and cannot be reduced. With binary scores and uniform difficulty, the total-variance factor is (1+2/K)/3: K=2 cuts it by 1/3, and K→∞ cuts it by at most 2/3. My arithmetic: K=8 achieves 87.5% of the attainable reduction, K=16 94%, K=32 97%. Worked example (n=198): raising K from 1 to 10 lowers the MDE from 13.2% to 7.5%. Applying the toy parameters (omega^2=1/9, sigma^2=1/6) to 164 tasks gives a sampled-pass@1 MDE of about 14.6 pp at K=1, 8.5 pp at K=8, 7.9 pp at K=16, and a floor of 7.3 pp. Our +3 pp sampled gain is therefore not detectable unless our measured omega^2 is much smaller. To compute the SE, average the K samples per task first and take the SE over tasks, which is equivalent to clustering on the task. Treating the n×K samples as independent understates the SE. Clustered SEs were 3.05× the naive ones on DROP, 1.88× on MGSM and 1.10× on RACE-H. Next-token-probability scoring, which removes sigma^2, does not apply to executed code generation. Beam-8 is near-deterministic, so resampling matters only for avg@n and pass@k.
  SRC: https://arxiv.org/abs/2411.00640 https://anthropic.com/research/statistical-approach-to-model-evals https://arxiv.org/abs/2607.04429
  EV: Verifiers re-derived the 1/3 and 2/3 figures and recomputed the MDE as 13.27% → 7.57%. The Anthropic post recommends resampling CoT answers and feeding per-question means into the SE, and says Inspect's 'epochs' parameter does this. Miller's Table 4 gives the clustered/naive ratios. evalci's cluster_ci resamples whole clusters and cites the >3× figure. Caveats: >3× is the worst case (DROP, where many questions share a passage), and the median ratio across the three evals is about 1.9×. The next-token-probability result is limited to non-CoT, first-token answers. The 14.6/8.5/7.9 pp GG numbers are my extrapolation from toy parameters; the real ω^2 and σ^2 must be measured from our own samples.

- high | Do not report mean ± 1.96·SE on 164-, 94- or 65-task benchmarks. CLT intervals under-cover at small N: nominal 95% intervals reach only about 92.5% coverage at N=100 (averaged over uniform θ), and coverage worsens at N=30, 10 and 3. For one arm's accuracy, use the Wilson score interval, (p̂ + z²/2n ± z·sqrt(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), or the Bayesian credible interval from Beta(1+successes, 1+failures) under a uniform Beta(1,1) prior. In the simulations these were the only methods at nominal coverage for small N. Clopper-Pearson is valid but too wide. For a paired difference (e.g., base vs. GRPO on the same 164 tasks), use the paired Bayes method. It gives the posterior of θ_A − θ_B by importance sampling (10k prior draws; θ ~ Beta(1,1); correlation via a Gaussian copula with ρ = 2ρ̂−1, ρ̂ ~ Beta(4,2)) and is narrower than an unpaired analysis. The tested non-Bayesian methods (paired CLT, unpaired CLT, bootstrap) all under-cover at small N.
  SRC: https://proceedings.mlr.press/v267/bowyer25a.html
  EV: This is a peer-reviewed ICML 2025 position paper (spotlight), and verifiers matched every number and recommendation to the PMLR PDF. A verifier independently computed exact Wald coverage: 0.922 averaged over θ~U(0,1) at N=100, 0.937 at N=164, p=0.87, 0.929 at N=65, p=0.87, and 0.949 at p=0.5. The shortfall at our accuracy levels is therefore real but modest (1-2 pp). This agrees with the classical literature (Brown, Cai & DasGupta 2001; Agresti & Coull 1998). Caveats: the main simulation draws true θ from the same Beta(1,1) prior the Bayesian method assumes, although mismatched-prior ablations still favour Bayes. No exact or score-based paired frequentist intervals were tested (exact McNemar, Newcombe hybrid score, Tango). At N=100 the coverage of paired CLT is within about 0.005-0.01 of paired Bayes, so the practical gain at 164 tasks is small. The intervals assume one binary outcome per task; avg@n data are clustered.

- high | Compare arms on the same tasks with a paired per-task analysis, and report the mean difference, its SE, a small-N-valid CI, and the per-task score correlation between the two arms. Pairing removes the variance from task difficulty, and the gain grows with that correlation (0.3-0.7 for frontier models on popular evals; measure ours). Defaults by metric. Beam-8 (one binary outcome per task): run the exact McNemar test, which equals the exact two-sided binomial sign test on the discordant counts (b up, c down): p = 2·Σ_{i≤min(b,c)} C(b+c, i)/2^{b+c}. evalci switches to the exact form below 25 discordant pairs. For our HumanEval 16 up / 1 down, p = 2·18/2^17 = 0.000275. avg@n or multi-seed: average each task over samples (and over training seeds), then run a paired test on the per-task differences. Options are the sign-flip permutation test (all 2^n patterns are enumerated exactly when n≤12; Monte Carlo otherwise), the null-shifted paired bootstrap over tasks, or, as Hochlehnert et al. 2025 do, paired t-test plus Wilcoxon signed-rank against the base model, reported at p<0.01 and p<0.001. Do not compare mean accuracies alone.
  SRC: https://anthropic.com/research/statistical-approach-to-model-evals https://arxiv.org/abs/2607.04429 https://arxiv.org/html/2504.07086v2 https://arxiv.org/abs/2411.00640
  EV: The Anthropic post recommends paired differences and the four reported quantities, citing a 0.3-0.7 correlation; Var(A−B)=Var(A)+Var(B)−2Cov(A,B). evalci Sec. 3.2 documents the exact-McNemar cutoff, exact enumeration for the sign-flip test, and the null-shifted bootstrap. That exact McNemar equals the exact sign test is a standard identity, and the p-value recomputes to 0.000275. Hochlehnert Sec. 4.2 describes per-problem averaging over seeds and then one-sided paired t-test/Wilcoxon; its tables show only the t-test stars. Caveats: evalci is a single-author, unrefereed preprint (July 2026), not on PyPI. Hochlehnert's tests are one-sided against each model's own base model; two-sided tests are the safer default unless the direction is pre-registered (my judgment). A separate claim with illustrative SE_paired numbers (e.g., a 0.64 HumanEval correlation from fictional models) was refuted 0-3, so do not cite those numbers.

- medium | Correct for multiple comparisons whenever several arms (GRPO variants, SFT, data/curriculum arms) are tested on one task set. In a simulation (4 models, 400 shared questions, ρ=0.5, true null, 2,000 replications), six uncorrected pairwise z-tests gave a family-wise error rate (FWER) of 0.192. Bonferroni gave 0.042 and an omnibus repeated-measures ANOVA gave 0.052. In a real example (6 models, 1,497 MMLU-Pro questions, 15 pairs), 5 pairs were nominally significant and only 3 survived Bonferroni/Holm. Suggested default (my synthesis): pre-declare the family per benchmark, e.g. each arm vs. the GRPO baseline and each arm vs. SFT, and apply Holm. A test that clears Holm is unlikely to be a multiplicity artifact. The HumanEval 16/1 result (p=0.000275) would survive even a Bonferroni family of 20 (0.000275×20 = 0.0055).
  SRC: https://arxiv.org/abs/2608.22659
  EV: The numbers match arXiv 2608.22659 (Mandujano Reyes, v2 Oct 2026). A verifier reproduced the FWER from first principles: under exchangeability, the studentized range of 4 normals gives about 0.204 uncorrected and about 0.041 Bonferroni, for any n. Medium rather than high because the source is a single-author, unrefereed preprint and the real-data claim was split 2-1. In that example, pairing did not change which pairs were nominally significant (5/15 either way); the evidence that pairing matters comes from the omnibus ANOVA. The count of survivors is also method-dependent: a mixed-effects model kept only 2. Holm itself was not simulated. A claim that one LMM recovers Miller's paired and clustered estimators was refuted 1-2.

- medium | Number of eval samples for sampled metrics. Hochlehnert et al. 2025 find that bootstrapping over about 30 eval runs stabilizes pass@1 on small benchmarks. They recommend at least 30 sampling seeds for small benchmarks (AIME, 30 problems), but used 10 for AIME'24/'25 and AMC'23 and 3 for larger sets such as MATH500, calling this a practical compromise. These are eval sampling seeds, not training seeds. For GG (my inference), this applies to avg@n and pass@k only. Combined with Miller's (1+2/K)/3 curve, K=16 samples per task captures about 94% of the attainable variance reduction for HumanEval-C, and K=32 is the conservative choice when cheap. Beam-8 needs no resampling.
  SRC: https://arxiv.org/html/2504.07086v2 https://arxiv.org/abs/2411.00640
  EV: Verifiers matched Takeaway 3 (Sec. 3.2.1), the Sec. 4.1 recommendation and the Sec. 4.2 practice in the v2 HTML. The 30-run figure is an empirical bootstrap judgment from 30-problem AIME scored with high-variance reasoning models, not a derived rule, so its transfer to 164-task code benchmarks is uncertain. A related claim that single-seed pass@1 has a 5-15 pp SD on AIME/AMC was refuted 0-3, so the magnitude of eval-seed variance on our benchmarks is unverified.

- medium | Training-seed variance vs. test-set sampling noise. The only verified evidence comes from pretraining: 10 Llama-2-7B runs that differ only in initialisation seed, with 21 checkpoints each. HumanEval seed SD is 1.11 pp, against an analytic 95% CI half-width of 3.98 pp, where CI = 1.96·sqrt(S(1−S)/N). On all 14 benchmarks studied, seed SD is below the CI: the ratio is 0.28 for HumanEval and about 0.8 for MMLU, BBH and MATH. Small benchmarks (HumanEval, COPA) show high variance on both measures. My inference: at 164 tasks, test-set sampling noise probably dominates training-seed noise, but this has not been shown for SFT or RLVR fine-tuning. There is no verified evidence on GRPO run-to-run variance at 1-7B, so we need to measure it ourselves.
  SRC: https://arxiv.org/html/2406.10229v1
  EV: Verifiers matched the Table 1 rows (HumanEval: size 164, mean 11.89, E(S,M) 1.11, CI 3.98) and the sentence 'Generally, the 7B seed variance is well below the 95% CI'. Limits: this is a single preprint (Meta, June 2024) with only 10 seeds and pretraining seeds rather than post-training seeds. The models are weak (about 12% HumanEval, and the averaged CI implies S≈0.07-0.08 across early checkpoints). The SD is averaged over checkpoints rather than taken at the final one, and the seed SD and the CI measure different noise sources.

- medium | Do not use IRT-based subsets (tinyBenchmarks IRT/IRT++, 100 items) to compare models. They sharply raise seed variance: SD goes from 0.80 to 1.80 on ARC-C, 0.41 to 1.16 on GSM8k and 0.21 to 2.06 on Hellaswag. They also lower monotonicity (GSM8k mon_disc 0.74 → 0.32). Across 70 models, Kendall τ can fall to 0.76, with about 12% of pairwise orderings flipped. IRT estimates average performance reasonably well, but not differences between models. For GG (inference), this argues against subsampling HumanEval-C or BringUp to save compute. IRT as a tool for analysing task difficulty was not evaluated.
  SRC: https://arxiv.org/html/2406.10229v1
  EV: Verifiers matched Table 3 (seed SD, Full/IRT/IRT++), Table 4 (monotonicity) and the verbatim caution. The setting is pretraining ablations on MCQ and GSM8k, not post-training on code, and the evidence is a single source.

- medium | Bayesian evaluation with closed-form posteriors (Hariri et al., 'Don't Pass@k', ICLR 2026). Each task's outcome is modeled as categorical with a uniform Dirichlet prior (Beta(1,1) when binary). With M tasks, N trials per task, C+1 categories with weights w_j, D prior columns, and ν = pseudo-counts + observed counts: T = 1 + C + D + N; μ = w0 + (1/(M·T)) Σ_α Σ_j ν_αj (w_j − w0); σ² = (1/(M²(T+1))) Σ_α [Σ_j (ν_αj/T)(w_j − w0)² − (Σ_j (ν_αj/T)(w_j − w0))²]. In the binary case with no prior data (my reduction), p_α = (1+s_α)/(N+2), μ = mean_α p_α, and σ² = Σ_α p_α(1−p_α) / (M²(N+3)). With a uniform prior, equal N and the same tasks, ranking by μ gives the same order as ranking by avg@N. Limitation: σ is conditional on the fixed M tasks. It leaves out task-sampling and training-seed variance, so it does not replace paired or clustered tests over tasks.
  SRC: https://arxiv.org/abs/2510.04265
  EV: Verifiers matched Algorithm 1 and Appendix A Eq. 15 and the Appendix B ranking-equivalence proof (μ = A + [N/(1+C+N)]·a), and confirmed ICLR 2026 acceptance. This is a single source. Two companion claims from the same paper were refuted: the z-threshold decision rule (1-2) and the claim of faster Kendall-τ convergence than pass@k (0-3). So use only the posterior formulas, not the tie rule or the convergence claims.

- medium | Recommended default protocol for GG, combining the verified findings above. The GG-specific numbers are my own derivations. (1) Headline metric beam-8 pass@1: per arm, report the accuracy with a Wilson or Beta(1+s,1+f) interval. Per comparison, report b/c discordant counts, Δ, and the exact two-sided McNemar/sign-test p-value, with a paired-Bayes (or exact/score) CI on Δ. (2) avg@n: K=16 samples per task, averaged per task. Then a paired sign-flip permutation test or paired bootstrap over tasks, with the task as the cluster, reporting the per-task correlation. (3) Multiple training seeds: average each task over seeds before the paired test (Hochlehnert-style), and report per-seed numbers and the seed SD. (4) Multiplicity: Holm within a pre-declared family per benchmark. (5) Power: state the MDE per benchmark (HumanEval-C ≈7 pp, dev ≈9 pp, clean BringUp ≈12 pp) and describe nulls below the MDE as inconclusive. (6) Report BringUp on the 53 clean programs as the headline; given its ≈12 pp MDE, treat it as supporting evidence. (7) Do not use IRT subsets.
  SRC: https://arxiv.org/abs/2411.00640 https://proceedings.mlr.press/v267/bowyer25a.html https://arxiv.org/abs/2607.04429 https://arxiv.org/html/2504.07086v2 https://arxiv.org/abs/2608.22659 https://arxiv.org/html/2406.10229v1
  EV: Each item traces to one of the verified findings above. The specific defaults (K=16, Holm, reporting on the clean subset, the MDE values) are synthesis and inference, not claims a source states for code benchmarks. The number of training seeds, the equivalence margin for the flat dev result, and the method for decontaminating or building a new held-out set are not backed by any verified claim in this run.

== caveats
"Coverage gaps: this run verified 19 claims, all on research questions 1 and 2 (tests, intervals, power, resampling, seed variance). Nothing passed verification for research question 3 (checkpoint selection, fairness of tuning budgets, compute-matched reporting), research question 4 (decontamination, time-split or LiveCodeBench-style held-out sets) or research question 5 (what A* RLVR/code papers report and what reviewers criticized). The same holds for rliable (IQM, stratified bootstrap, performance profiles), Almost Stochastic Order, Dodge et al.'s expected max performance, TOST equivalence tests, pre-registration, Benjamini-Hochberg, the variance of the unbiased pass@k estimator (Chen 2021), and IRT for code difficulty. These are probably among the claims left unchecked when votes failed at the session limit, so recommendations in those areas are judgment, not verified evidence. Source quality: Miller 2024 and the Anthropic post are companion works by one author and form a single line of evidence. 2608.22659 (multiple comparisons) and 2607.04429 (evalci) are single-author, unrefereed 2026 preprints. Madaan 2024 is an unrefereed preprint on pretraining, not post-training. Bowyer (ICML 2025) and Hariri (ICLR 2026) are peer-reviewed. Several verifiers ran out of web-search budget, so contradicting work was searched for only partially. Transfer risk: the seed-variance evidence comes from 7B pretraining with weak models (about 12% HumanEval) and the eval-seed evidence from 30-problem math benchmarks. Neither shows RLVR training-seed variance on code. The MDE figures use the normal-approximation power formula, which Bowyer et al. warn is less reliable below a few hundred items. They assume our measured omega^2\u22480.095 carries over to the dev set and BringUp, and the sampled-metric MDEs use Miller's toy sigma^2=1/6 and omega^2=1/9; all of these should be recomputed from our own per-task data. Refuted claims (do not cite): the illustrative SE_paired numbers and fictional correlations in Miller's Table 5, the Bayes@N z-threshold tie rule and the faster-convergence claim, the claim that one LMM recovers Miller's estimators, the 5-15 pp single-seed SD on AIME/AMC, and the claim that greedy decoding is as variable as T=0.7 (ReasonBench). Time sensitivity: three sources are 2026 preprints, and 2608.22659 already has a v2 (Oct 2026), so their numbers may still change."
== openQuestions
[
 "How large is GRPO run-to-run (training-seed) variance at 1-7B on HumanEval-C beam-8 and avg@n compared with the about 7 pp test-set MDE? None of the verified evidence covers RLVR. Before setting the seed count per arm (at about 8 GPU-hours per seed), a small pilot (e.g., 3 seeds each for the headline GRPO arm and its baseline) is needed to estimate the seed SD.",
 "How should a multi-seed, multi-sample paired comparison be modeled: per-task averaging followed by a paired permutation test (Hochlehnert-style), or a hierarchical seeds \u00d7 tasks \u00d7 samples model? The one mixed-model claim was refuted 1-2, so no hierarchical estimator is verified for our design.",
 "How should the flat dev beam-8 result be reported? Showing that it is equivalent rather than inconclusive needs a TOST or non-inferiority test with a stated margin and exact small-N methods at n=94, and neither TOST nor margin choice was verified.",
 "What contamination-robust evaluation will A* reviewers accept for C transpilation? Reporting on the clean subset alone leaves BringUp at a \u224812 pp MDE. Options include a new held-out or time-split C test set built from real I/O tests, or a documented decontamination method; none were covered by verified claims, nor were reviewer expectations on checkpoint selection and equal tuning budgets for baselines."
]
== refuted
[
 {
  "claim": "When two models are compared on the same questions, a paired analysis of per-question differences lowers the variance of the estimated difference by 2Cov/n. The formula is SE_paired = sqrt(SE_A^2 + SE_B^2 - 2*SE_A*SE_B*Corr(s_A,s_B)). Question-level scores tend to be positively correlated, even across unrelated models, so pairing is a 'free' gain in precision. In the paper's uniform-score example with correlation 0.5, pairing cuts the estimator variance from 1/6 to 1/9. Table 5 uses illustrative correlations from fictional models, such as 0.64 on HumanEval; these are not measured values.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2411.00640"
 },
 {
  "claim": "Decision rule for comparing two models: compute z = |mu - mu'| / sqrt(sigma^2 + sigma'^2) and ranking confidence rho = 0.5(1 + erf(z/sqrt(2))). Two models are declared tied when z is below the threshold. The default threshold is z = 1.645, which the paper maps to rho = 0.95. The paper's z-to-confidence labels are loose: it also says z = 1.96 corresponds to 97.5%.",
  "vote": "1-2",
  "source": "https://arxiv.org/abs/2510.04265"
 },
 {
  "claim": "On AIME'24, AIME'25, HMMT'25 and BrUMO'25 (20 models; gold-standard ranking from N = 80 trials), the Bayes@N ranking reaches Kendall tau > 0.90 against the gold standard by N = 10 trials and approaches tau of about 1 near N = 80. AIME'25 is the exception and plateaus at tau of about 0.95. Pass@k variants start lower and converge more slowly on all four datasets. Mean convergence trial: about 44 for Bayes@N vs about 70 for the best Pass@k on HMMT'25, and about 27 vs about 49 on BrUMO'25. Note that these are math benchmarks of about 30 questions each, not code benchmarks.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2510.04265"
 },
 {
  "claim": "A single linear mixed model (model as fixed effect, cluster and question-within-cluster as random effects: s_{i,c,m} = mu + alpha_m + u_c + delta_{i,c} + eps, fit as lmer(score ~ model + (1|cluster) + (1|cluster:question))) recovers Miller's paired and clustered estimators at K=2. It extends them to any K and to unbalanced, clustered designs.",
  "vote": "1-2",
  "source": "https://arxiv.org/abs/2608.22659"
 },
 {
  "claim": "On small reasoning benchmarks (AIME'24 with 30 problems, AMC'23 with 40), pass@1 measured from a single sampling seed has a standard deviation of 5 to 15 percentage points across seeds. One question flipping moves pass@1 by 2.5 to 3.3 pp.",
  "vote": "0-3",
  "source": "https://arxiv.org/html/2504.07086v2"
 },
 {
  "claim": "Run-to-run variance in LLM reasoning evaluation does not go away under greedy decoding. In ReasonBench, the inter-quartile range of scores across repeated runs at temperature 0 is comparable to, and in several cells wider than, the range at temperature 0.7. A beam or greedy metric that looks near-deterministic still needs multiple runs to show its variance.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2512.07795"
 }
]
== unverified
[]
== sources
[
 {
  "url": "https://arxiv.org/abs/2411.00640",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://proceedings.mlr.press/v267/bowyer25a.html",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2510.04265",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2608.22659",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2607.04429",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://anthropic.com/research/statistical-approach-to-model-evals",
  "quality": "primary",
  "angle": "Small-sample eval statistics (core methods)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2504.07086v2",
  "quality": "primary",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2406.10229v1",
  "quality": "primary",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://www.lesswrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evaluations-call-into-question-recent-llm",
  "quality": "blog",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2512.07795",
  "quality": "primary",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://ar5iv.labs.arxiv.org/html/2304.01910",
  "quality": "primary",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2605.28022",
  "quality": "primary",
  "angle": "RLVR seed and decoding variance",
  "claimCount": 5
 },
 {
  "url": "https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/",
  "quality": "primary",
  "angle": "Aggregate comparison, power and multiplicity",
  "claimCount": 5
 },
 {
  "url": "https://www.arxiv.org/pdf/2204.06815",
  "quality": "primary",
  "angle": "Aggregate comparison, power and multiplicity",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2411.00640",
  "quality": "primary",
  "angle": "Aggregate comparison, power and multiplicity",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/1709.09500",
  "quality": "primary",
  "angle": "Aggregate comparison, power and multiplicity",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2403.04811",
  "quality": "primary",
  "angle": "Contamination-robust code evaluation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2403.07974",
  "quality": "primary",
  "angle": "Contamination-robust code evaluation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2311.04850",
  "quality": "primary",
  "angle": "Contamination-robust code evaluation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2504.13837v4",
  "quality": "primary",
  "angle": "Reviewer critiques and reporting norms (skeptical)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2504.07086v2",
  "quality": "primary",
  "angle": "Reviewer critiques and reporting norms (skeptical)",
  "claimCount": 5
 },
 {
  "url": "https://icml.cc/virtual/2026/poster/61082",
  "quality": "primary",
  "angle": "Reviewer critiques and reporting norms (skeptical)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2507.10532v3",
  "quality": "primary",
  "angle": "Reviewer critiques and reporting norms (skeptical)",
  "claimCount": 5
 },
 {
  "url": "https://aclanthology.org/D19-1224/",
  "quality": "primary",
  "angle": "Reviewer critiques and reporting norms (skeptical)",
  "claimCount": 5
 }
]
== stats
{
 "angles": 5,
 "sourcesFetched": 24,
 "claimsExtracted": 120,
 "claimsVerified": 25,
 "confirmed": 19,
 "killed": 6,
 "unverified": 0,
 "afterSynthesis": 10,
 "urlDupes": 0,
 "budgetDropped": 6,
 "agentCalls": 106
}
