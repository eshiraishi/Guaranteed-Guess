# Evidence: 03-curriculum (deep-research output, Oct 24, 2026)

Raw synthesis of the 3-vote verified claims. Fields: confidence | claim, SRC (sources), EV (evidence and vote). Sections after the findings: caveats, open questions, refuted claims, sources, stats.

SUMMARY: The 20 verified claims come from math, puzzle and reasoning RLVR on 1B-8B models, mostly Qwen2.5, with single seeds and no code tasks. They point to one mechanism. Prompts at pass rate 0 or 1 give no gradient, and choosing prompts by their online pass rate near 0.5 (learnability p(1-p)) gives small gains (about 2-4 pp average) and saves compute (1.5-3.3x fewer steps, or 2-3.35x fewer rollouts at matched accuracy). Four methods from different groups do this selection: band filtering (Bae), per-step rejection sampling (LILO), a bandit over categories (SEC) and target-difficulty tracking (AdaRFT); GRESO adds skipping before rollout. ScaleRL's 8B leave-one-out ablations show that this kind of curriculum and zero-variance filtering change compute efficiency, not the performance ceiling. A fixed easy-to-hard order often did worse than random sampling. For prompts the policy never solves, which is our BringUp case, reordering existing prompts cannot help. Three separate groups (R3, QuestA, POPE) instead report gains from starting rollouts from a prefix of a reference solution and removing the prefix at test time. QuestA's own ablation, however, credits most of its gain to RL on hard-filtered data and only about 3 pp to the hint. No verified claim covers SFT curricula, static program-difficulty predictors, function-to-whole-program or O0-to-O2 schedules, or long-context curricula, so research questions 2-4 are still open. What this means for GG, as an untested extrapolation: use the pass rate already available from each group of 8 to skip or drop zero-variance MBPP-C prompts, which saves compute. For BringUp, try compiler-generated ARM prefixes as POPE-style guidance, trained in their own phase or with only hard prompts.

- high | Prompts where every rollout in the group gets the same reward (all-pass or all-fail) give zero advantage, so they add no policy gradient under GRPO or RLOO. The useful signal per prompt is learnability, the binary-reward variance p(1-p), estimated as p-hat(1-p-hat), optionally times K/(K-1). For RLOO, PPO and VinePPO, and for Dr. GRPO-style advantages without std division, the improvement bound grows linearly with p(1-p). Std-normalized GRPO removes this dependence in theory, and its expected |advantage| is 2*sqrt(p(1-p)), which still peaks at p=0.5. Even so, zero-advantage prompts still dilute the update when a KL term or a token-level or batch-level loss normalizer counts them. That is why ScaleRL computes the loss on the 'effective batch' only.
  SRC: https://arxiv.org/abs/2502.12272 https://arxiv.org/abs/2505.14970 https://arxiv.org/abs/2510.13786
  EV: LILO Sec 2.2 and Theorem 3.1: zero learnability gives A_t=0 and 'no contribution to the model update'. Normalization 'removes the dependency' on learnability, and the dependency returns without the std division (the paper cites Dr. GRPO). LILO's own Table 5 still finds a correlation between learnability and ||grad||^2 under normalized GRPO on MATH (r=0.384), so 'removes' holds only in theory. SEC Appendix A derives E|A| = 2*sqrt(p(1-p)), and a verifier re-derived it. ScaleRL defines the effective batch as prompts with non-zero variance and adopts it. These are three independent primary sources, and the theoretical claims are consistent across them. For GG: our arms differ in normalization (DeepSeekMath GRPO uses std, Dr. GRPO does not, TRL default), so in theory p(1-p) selection should matter more for the Dr. GRPO arm.

- medium | In math RLVR, choosing prompts whose current-policy pass rate is in a band around 0.5 does better than uniform sampling, by small margins. Removing only the all-pass and all-fail prompts (DAPO-style 0<p<1) gives a smaller gain. One-sided filters (only easy or only hard) do worse than plain GRPO.
  SRC: https://arxiv.org/abs/2504.03380 https://arxiv.org/abs/2502.12272
  EV: Bae et al. (EACL 2026) estimate p from 16 rollouts after an SFT cold start, which resembles GG's SFT start. Qwen2.5-3B averaged 26.3 with plain GRPO and 30.1 with 0.3<p<0.7, and every symmetric band scored about 28-30. The paper prints 27.3 for the (0,1) row, but its own numbers average to 28.1. Skewed one-sided filters scored 24.2-25.9. Qwen2.5-7B went from 39.6 to 41.8 (AMC 42.2 to 47.0), against 39.9 for 0<p<1. Not every benchmark improves: 3B AMC fell from 30.1 to 28.9. LILO reaches the baseline's best accuracy 1.5-3.3x sooner, with +1.6 to +2.7 pp at the end. Its GRPO arm (Qwen2.5-1.5B, ORZ57K) shows the smallest effect: 1.5x, 35.5 to 37.1, and no AIME gain. The direction agrees across groups, but every result is a single seed with no confidence intervals, and AIME has only 30 problems (6.6 vs 3.3 is 2 vs 1 problems). Bae's bound on expected improvement, stated as the variance p(1-p), failed verification 0-3, so only the empirical result should be cited. Note for GG with G=8: the 0.3-0.7 band means 3-5 correct out of 8, and the standard error of p-hat at p=0.5 is about 0.18, so a band picked from one step is noisy.

- medium | LILO's schedule runs at every step and keeps no state across steps. It samples a candidate pool of 4x the batch size, rolls out 8 attempts per prompt, keeps the top-|B| prompts by p-hat(1-p-hat) and reuses those rollouts for training. Sampling costs about 4x. When train accuracy nears 95%, the pool must grow to 8x. A buffer refreshed only every 5-50 iterations overfitted badly.
  SRC: https://arxiv.org/abs/2502.12272
  EV: Algorithm 2 and Sec 5: |D|=4x|B| and N_learnability=8 'works well with minimal overhead'. VinePPO on GSM8K near 95% train accuracy needed 8x. Table 1 on TPUs: PPO took 220s per iteration and PPO+LILO 280s. The cost is low there only because sampling on TPU takes 20s of the 220s. With vLLM on GPUs and GG prompts of up to 23k tokens, rollouts dominate, so 4x sampling would roughly multiply wall-clock. In the Appendix Fig 11 buffer ablation (50% of the batch from the buffer), overfitting stopped only when the buffer was refreshed every step. This is a single paper and one small ablation of an SFL-style mix. GRESO and ScaleRL keep state across steps without reported harm, so the buffer result should not be read as general.

- medium | GRESO decides which prompts to skip before rollout, using each prompt's recent history of zero-variance groups. It skips a prompt with probability 1 - p_e^z, where z is the number of consecutive recent zero-variance groups. It keeps separate exploration probabilities p_e for easy and hard prompts: initial 50%, minimum 5%, moved in 1% steps toward a 25% target zero-variance ratio split 1:2 easy:hard. It matched DAPO dynamic sampling with 2.1-3.35x fewer rollouts.
  SRC: https://arxiv.org/abs/2506.02177
  EV: Eq. 3, Algorithm 1 and Appendix E give these settings. Qwen2.5-Math-7B on DAPO+MATH went from 13.1M to 6.3M rollouts (about 2.1x) at 57.5 vs 57.8 average. The 3.35x maximum is 7B on Open-R1-30k (11.4M to 3.4M). 1.5B: 48.5 vs 48.4 average with 3.3M vs 7.6M rollouts. Rollout was up to 2.4x faster and total training up to 2.0x faster. Some cells drop (7B AIME24 34.2 to 32.5), within AIME noise. It used group size 8, as our run does, but math only and a single paper. The paper's supporting statistic that over 90% of zero-variance prompts were already flagged in earlier epochs, and about 20% later become useful again, failed verification 0-3. So the temporal-stability premise is plausible but unconfirmed.

- medium | At 8B scale in math RLVR, curriculum and zero-variance filtering mainly change compute efficiency (exponent B and midpoint C_mid), not the asymptotic ceiling A. ScaleRL's curriculum permanently drops prompts whose historical pass rate is 0.9 or higher (No-Positive-Resampling, after Polaris / An et al. 2025). ScaleRL's 'effective batch' computes the loss only over prompts with non-zero variance.
  SRC: https://arxiv.org/abs/2510.13786
  EV: Leave-one-out (LOO) ablations at 8B dense, 16k GPU-hours, Polaris-53k data. Every variant's A falls in 0.590-0.610, inside the paper's ±0.02 seed band. Removing the curriculum (LOO-uniform-sampling) cut B from 1.92 to 1.62, the largest drop of any variant, while A moved only from 0.610 to 0.605. LOO-batch (no zero-variance filtering) had A=0.595 and B=1.69, against 2.01 for full ScaleRL. The forward ablations claim A gains of 0.535 vs 0.520 (curriculum) and 0.540 vs 0.520 (effective batch), but both are inside the ±0.02 noise band, so the paper's wording is slightly inconsistent. Only one curriculum type, easy-prompt removal, was tested, with one seed per LOO run and A extrapolated from fitted curves. A separate claim about the details of No-Positive-Resampling failed verification 0-3. The ≥0.9 rule here comes from verifier evidence on the confirmed claims, so cite it as reported, not as independently confirmed. For GG: curricula on MBPP-C should save compute, but this evidence gives no reason to expect them to unlock BringUp.

- medium | A fixed easy-to-hard ordering often scored worse than uniform random sampling, and a hard-to-easy ordering scored much worse. An adaptive curriculum, SEC, beat random sampling on most out-of-distribution (OOD) splits at 3B and gave mixed results at 7B. SEC treats curriculum choice as a non-stationary bandit over categories such as difficulty levels. Each arm's reward is the batch-mean |advantage|, which tracks p(1-p).
  SRC: https://arxiv.org/abs/2505.14970
  EV: GRPO on Qwen2.5-3B/7B with Countdown, Zebra, ARC-1D and math tasks, no code. Ordered beat Random in 7 of 18 Table 1 cells. Countdown 3B: Ordered 0.551 ID / 0.321 OOD vs Random 0.859 / 0.479. Countdown 7B: 0.820 / 0.442 vs 0.858 / 0.566. Ordered won both Zebra 3B splits and 3B ARC-1D OOD. For the reverse curriculum, Fig 1 shows only that it 'severely limits' performance, with no numbers. SEC updates arm values by EMA (Q_{t+1}=α r + (1-α)Q) and samples arms by Boltzmann softmax over Q/τ, with α 0.2-0.5 and τ 0.2-1.0, tuned per task. It reuses GRPO's advantages, so it needs no extra rollouts, but the paper gives no overhead numbers. SEC vs random at 3B: Countdown OOD 0.479 to 0.542, Zebra OOD 0.285 to 0.345, ARC-1D OOD 0.313 to 0.381, AIME24 0.075 to 0.100. At 7B: AIME24 0.138 to 0.175 and AMC 0.486 to 0.511, but SEC lost Countdown OOD (0.555 vs 0.566), ARC-1D OOD (0.418 vs 0.436) and MATH500 (0.761 vs 0.774). SEC also helped under PPO and RLOO, on 3B Countdown only. One lab ran both its method and the baselines; results are best checkpoints with no seeds.

- medium | AdaRFT moves a scalar target difficulty T with T' = clip(T + 50*tanh(2*(R_avg - 0.5)), 0, 100). Each batch is the 1024 problems whose fixed offline difficulty is nearest T. The offline difficulty is a reference model's pass rate over 128 samples. Gains are about 2-4 pp in short single-seed runs. The headline 'up to 2x' speedup leaves out the cost of annotating difficulty.
  SRC: https://arxiv.org/abs/2504.05520
  EV: Difficulty is d_i = 100(1 - s_i/128), scored once by Qwen2.5-Math-7B; T starts at 0. Results with PPO, about 100 steps on 8 A100s, one seed per configuration. Qwen2.5-Math-1.5B on skew-difficult data: 37.41 to 40.48, and PPO needs 43 more steps (+71.7%) to match AdaRFT at step 60. On uniform data: 37.20 to 41.11. Qwen2.5-7B: 44.17 to 46.83, with +24 / +25 steps for PPO / PPO on filtered data. The authors call these 'empirical trends rather than statistically significant'. They put the upfront annotation at 1.28M scorer rollouts for 10k problems at pass@128, and say pass@40-64 keeps most of the ordering. For GG: GG-scored pass rates already in our eval and GRPO-run-1 logs, if kept per prompt, would serve as a free difficulty annotation. A fresh pass@128 on 23k-token prompts would be expensive.

- high | For prompts the policy (almost) never solves, three independent groups report that RL with prefixes of a correct reference solution makes them learnable, and that the gain carries over to unguided prompts at evaluation. The increment over plain RL on the same hard-filtered data is modest.
  SRC: https://arxiv.org/abs/2402.05808 https://arxiv.org/abs/2507.13266 https://arxiv.org/abs/2601.18779
  EV: R3 (Llama2-7B, PPO, 8 reasoning QA tasks): rollouts start from intermediate states of a correct demonstration, giving +4.1 over RL and +5.4 over SFT on average. Its final schedule mixes start points from all stages, because a strictly staged end-to-start schedule lost performance at each stage change. QuestA (OpenMath-Nemotron-1.5B, GRPO): prepends the first p fraction of a reference solution to hard prompts (100 steps at p=50%, then 1900 at 25%) and removes hints at evaluation. AIME24 went from 61.77 to 72.50 and the five-benchmark average from 53.14 to 63.26. In the no-hint ablation on the same data, plain RL reached 69.48 / 59.79 on AIME24/25, so the hint adds about 2.5-3 pp. The hinted run matched the no-hint 2K-step result in about 1.1K steps. 'Near-zero pass rate' is loose: the second filter keeps hinted prompts with 0-4 of 8 correct. POPE (Qwen3-4B-Instruct-2507, GRPO): for each hard prompt it tries evenly spaced prefixes and keeps the shortest one that gives a success. If none succeeds, it uses a random prefix shorter than 1/4 of the solution. It trains on a 1:1 mix of guided and unguided prompts, with a 'continue from the partial solution' instruction. Hard prompts were those with 0 of 128 correct at a 32k-token budget. On those hard training prompts, pass@16 went from 32.89 (hard-only RL) to 42.53 (base model 7.42). Held-out gains were smaller: AIME25 pass@1 49.58 to 53.12, HMMT25 31.04 to 37.81. All three are math or reasoning QA, single runs, unreplicated. The direction is consistent, but the magnitude is uncertain. Separate BREAD claims (expert-prefix branching, per-prompt binary search for the prefix) failed verification 0-3 and should not be cited. For GG, transpilation has an unusually cheap oracle: the compiler's ARM output for the same C source.

- low | Mixing unguided easy problems into RL on zero-pass problems made learning the hard problems much worse ('ray interference'), even though benchmark scores rose. Prefix guidance kept hard-set progress when easy problems were mixed in.
  SRC: https://arxiv.org/abs/2601.18779
  EV: POPE Table 2: hard-only RL scored 13.55 pass@1 / 32.89 pass@16 on the hard set. Hard plus 1K unguided easy problems scored 2.24 / 25.12, while AIME25 pass@1 rose from 49.58 to 61.88. Hard plus guidance plus 1K easy scored 13.98 / 36.42, with AIME25 at 62.01 / 84.31. With 256 easy problems the pattern was the same (8.22 / 23.81 unguided, 14.32 / 38.93 guided). This rests on one preprint and one model, and the hard set is the training problems themselves, not held-out ones. Dilution is not ruled out: hard prompts were only about 20% of the batch, and there was no compute-matched control. For GG: adding unguided BringUp prompts to MBPP-C batches would probably starve them. Guide them, or train them in their own phase.

- low | Starting schedule for GG, extrapolated from the sources above (untested in code or assembly). (1) For function-level RL on MBPP-C, keep G=8. Compute the loss only on prompts with non-zero variance. Optionally skip prompts GRESO-style before rollout (p_e starting at 0.5, minimum 0.05, step 0.01, 25% target split 1:2 easy:hard). Drop prompts whose historical pass rate is 0.9 or higher. Re-estimate pass rates every step from the rollouts already generated, never from a stale buffer. (2) For BringUp or whole programs, where the base model almost never passes, add POPE/QuestA-style prefixes of the compiler-generated ARM: try evenly spaced prefix fractions and keep the shortest that passes, or start at 50% and anneal to 25%. Train guided and unguided copies 1:1, in their own phase or with only hard prompts, and evaluate unguided. (3) Expect about 2-4 pp and 1.5-3x compute savings from (1), with no gain in ceiling. The only lever with evidence for zero-pass prompts is (2).
  SRC: https://arxiv.org/abs/2510.13786 https://arxiv.org/abs/2506.02177 https://arxiv.org/abs/2502.12272 https://arxiv.org/abs/2601.18779 https://arxiv.org/abs/2507.13266 https://arxiv.org/abs/2504.03380
  EV: This is an extrapolation that combines the findings above; no source tested it on code. Points of tension: LILO found periodic buffers overfit, while GRESO and ScaleRL use history across steps successfully. Bae used 16 rollouts per prompt, but our 8 make the band estimate noisy, which favors dropping only all-pass and all-fail prompts (0<p<1) over a narrow 0.3-0.7 band. Prefixes add tokens to prompts that already reach 23k+, so they may hit context limits.

== caveats
"Domain and scale: all 20 verified claims come from math, synthetic puzzles (Countdown, Zebra, ARC-1D) or reasoning QA, on 1B-8B models, mostly the Qwen2.5 family, with one Llama2-7B, one Rho-Math-1B and one Qwen3-4B. None tests code, compilers or assembly. GG's 1.5B scale matches, but its domain and its binary execution reward with long prompts do not.\n\nStatistics: almost every result is a single seed with no confidence intervals, and SEC and R3 report best checkpoints. Many gains fall within benchmark noise (AIME has 30 problems). ScaleRL's own \u00b10.02 seed band swallows its claimed asymptote gains. No specific number has been replicated by another group; each method is evaluated only by its authors, on baselines they chose.\n\nConfidence labels: 'high' marks a direction backed by several independent primary sources (the zero-variance and p(1-p) mechanism; reference-prefix guidance for zero-pass prompts). 'Medium' marks a single primary paper whose numbers were verified 3-0. 'Low' marks a single source with notable confounds (POPE's interference result: training-set metric, dilution) and my own extrapolated schedule.\n\nCoverage gap: this topic hit the session limit (about 25 votes failed). Many named methods have no verified claim in this set: DAPO dynamic sampling details, VCRL, Prompt Curriculum Learning, SPaRK, DOTS, Polaris, Skywork-OR1, AceReason-Nemotron math-then-code, the DeepScaleR 8k-16k-24k context curriculum, Kimi k1.5, ProRL, E2H Reasoner, StepHint and BREAD. Missing from this set does not mean missing evidence; those claims may still verify on resume.\n\nRefuted claims not to cite: Bae's bound on expected improvement stated as the variance p(1-p), GRESO's statistic that over 90% of zero-variance prompts were already flagged, the exact No-Positive-Resampling wording in ScaleRL, and both BREAD claims.\n\nResearch questions 2-4 are unanswered: SFT and NMT curricula (including Platanios's competence-based curriculum), static difficulty metrics or IRT as predictors of pass rate, and curricula from functions to whole programs, O0 to O2, or short to long context.\n\nSeveral verifiers ran out of web-search budget, so the search for contradicting evidence was incomplete.\n\nTime sensitivity: the sources date from 2024 to 2026. POPE (Jan 2026) is an un-peer-reviewed preprint. QuestA (v4, Mar 2026), AdaRFT (v4, Jun 2026) and LILO (v6, Apr 2026) were revised recently.\n\nWall-clock figures do not transfer: LILO's 220s to 280s is on TPUs, where sampling is cheap, and GRESO's figures are for short math prompts.\n\nBae's paper has a small arithmetic error: its (0,1) row averages 28.1, not the printed 27.3."
== openQuestions
[
 "Do cheap static features predict GG's per-prompt pass rate well enough to act as a prior before any rollouts? Candidates: x86 instruction count or prompt tokens, number of functions, loop nesting, SIMD/FP use, optimization level, call-graph depth. This can be tested in minutes against existing dev, HumanEval and BringUp eval logs; no verified literature claim covers it.",
 "Does guidance from prefixes of the compiler-generated ARM carry over to unguided whole-program translation, given prompts of 23k+ tokens? Would splitting BringUp programs into per-function tasks, each checked by execution, be a better route to a non-zero pass rate?",
 "Does p(1-p) prioritization help more under Dr. GRPO (no std division) than under DeepSeekMath GRPO, as LILO's theory predicts? Our existing GRPO arms would show this directly.",
 "Once the resumed verification finishes, do the claims not yet checked change these recommendations? These include DeepScaleR's context-length curriculum, AceReason's math-then-code ordering, Skywork-OR1/Polaris filtering, E2H Reasoner, and the BREAD and StepHint hint schedules."
]
== refuted
[
 {
  "claim": "Theory: in RLVR with binary rewards, the lower bound on expected policy improvement is the variance of task-level success probabilities p(1-p). This implies that prompts of intermediate difficulty (pass rate near 0.5) maximize learning per step, and that balanced filtering maximizes this bound.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2504.03380"
 },
 {
  "claim": "Whether a prompt is zero-variance (every rollout in the group gets the same reward, so the advantage is zero) stays fairly stable from one epoch to the next. The paper reports that over 90% of zero-variance prompts in an epoch had already been flagged in earlier epochs. About 20% of previously zero-variance prompts later become effective again. This was measured on math RLVR with Qwen2.5-Math 1.5B/7B and DeepSeek-R1-Distill-Qwen-1.5B, group size 8.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2506.02177"
 },
 {
  "claim": "ScaleRL's curriculum component (No-Positive-Resampling, taken from Polaris / An et al. 2025) permanently removes every prompt whose policy pass rate is at least 0.9 from all later epochs. The stated reason is that a prompt that becomes easy stays easy and no longer gives a useful gradient. In the forward ablation (8B dense, math RLVR) the authors report that this improves scalability and the asymptotic reward A, but the text gives no numeric change in A or B.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2510.13786"
 },
 {
  "claim": "On prompts where the small model alone has near-zero success, GRPO stalls because no group contains a correct trace. BREAD fixes this by branching rollouts from short expert-solution prefixes, so that each update has at least one successful trace. This directly targets zero-advantage, all-fail groups like BringUp whole programs.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2506.17211"
 },
 {
  "claim": "The hint length is not fixed. A per-prompt binary search over the expert solution (Episode Anchor Search) finds the shortest prefix that lets the policy succeed, and it adjusts the branch point as training goes on. Rollouts use 8 responses per prompt, the same group size as our GRPO run.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2506.17211"
 }
]
== unverified
[]
== sources
[
 {
  "url": "https://arxiv.org/abs/2504.03380",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.02177",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2502.12272",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2510.13786",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.14970",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2504.05520",
  "quality": "primary",
  "angle": "RLVR online difficulty filtering and learnability curricula (primary evidence)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2507.13266",
  "quality": "primary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.17211",
  "quality": "primary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2601.18779",
  "quality": "primary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2402.05808",
  "quality": "primary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://www.preprints.org/frontend/manuscript/1de44aa1704543cb7d6d074a83dedf59/download_pub",
  "quality": "secondary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2605.17174",
  "quality": "primary",
  "angle": "Hint, prefix and reverse-curriculum RL for prompts the base model never solves",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.22312",
  "quality": "primary",
  "angle": "Context-length and staged curricula in large reasoning RL recipes, plus null results",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.16400",
  "quality": "primary",
  "angle": "Context-length and staged curricula in large reasoning RL recipes, plus null results",
  "claimCount": 5
 },
 {
  "url": "https://docs.nvidia.com/nemo/rl/nightly/guides/grpo-deepscaler.html",
  "quality": "primary",
  "angle": "Context-length and staged curricula in large reasoning RL recipes, plus null results",
  "claimCount": 5
 },
 {
  "url": "https://aclanthology.org/N19-1119/",
  "quality": "primary",
  "angle": "SFT curriculum for code, translation, decompilation and assembly",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2108.06084",
  "quality": "primary",
  "angle": "SFT curriculum for code, translation, decompilation and assembly",
  "claimCount": 4
 },
 {
  "url": "https://arxiv.org/abs/2403.05286",
  "quality": "primary",
  "angle": "SFT curriculum for code, translation, decompilation and assembly",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2012.03107",
  "quality": "primary",
  "angle": "SFT curriculum for code, translation, decompilation and assembly",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2602.07882",
  "quality": "primary",
  "angle": "Predicting program difficulty: static metrics and IRT vs online pass rate",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2507.04632",
  "quality": "primary",
  "angle": "Predicting program difficulty: static metrics and IRT vs online pass rate",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2609.09075",
  "quality": "primary",
  "angle": "Predicting program difficulty: static metrics and IRT vs online pass rate",
  "claimCount": 5
 }
]
== stats
{
 "angles": 5,
 "sourcesFetched": 22,
 "claimsExtracted": 109,
 "claimsVerified": 25,
 "confirmed": 20,
 "killed": 5,
 "unverified": 0,
 "afterSynthesis": 10,
 "urlDupes": 1,
 "budgetDropped": 7,
 "agentCalls": 104
}
