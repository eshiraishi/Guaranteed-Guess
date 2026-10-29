# 03. Curriculum: difficulty-aware data scheduling for SFT and RL

**Status (Oct 24, 2026): plan.** Tags: [V] verified, [R] repo, [L] unverified literature, [P] proposal.

## The question, answered first
> "Can we use the difficulty of the program to do curriculum learning?"

- **As a prior, yes. As the schedule itself, the evidence says no.**
  - The signal that works is the **policy's own current pass rate** on each prompt. Prompts at pass rate 0 or 1 give
    zero gradient. Choosing prompts near p = 0.5 (learnability p(1−p)) gave about +2–4 pp and 1.5–3.3x fewer steps,
    or 2–3.35x fewer rollouts at matched accuracy [V, math/puzzles at 1–8B].
  - A **fixed easy-to-hard order often did worse than random sampling.** Ordered beat Random in only 7 of 18 cells, and
    hard-to-easy was much worse [V, SEC].
  - Static program difficulty should therefore set the **starting estimate** of a prompt's pass rate and define the
    **stages** below. It should not be a fixed ordering.
- **Curriculum mostly saves compute; it does not raise the ceiling.** In ScaleRL's 8B leave-one-out runs, curriculum
  and zero-variance filtering changed efficiency (B) far more than the asymptote (A, within ±0.02) [V].
- **For prompts the model never solves, such as BringUp in run 1, reordering cannot help.**
  - The lever with evidence is **reference-prefix guidance**: start rollouts from a prefix of a correct solution, and
    evaluate without it. Three independent groups support it (R3, QuestA, POPE) [V]. Most of QuestA's gain comes from RL
    on hard-filtered data, and the hint adds about 3 pp [V].
  - Transpilation has an unusually cheap source of correct prefixes: **the compiler's own target assembly** [P]. Using
    it fits the "no teachers" rule, because the compiler is not a model.
- **Our own run wastes most of its signal.** 57% of groups in run 1 were all-pass or all-fail [R]. Fixing that is the
  first and cheapest gain.

## 1. Layers, applied in order
Each layer is one switch, so each can be ablated against the layer below it ([06](06-schedule.md)).

### L0. Effective batch (zero-variance filtering)
- **What:** compute the loss only over prompts with a non-zero reward variance (ScaleRL's "effective batch") [V], and
  refill the batch to its full size.
  - **Post-rollout, DAPO-style:** oversample prompts and keep only those with mixed groups.
  - **Pre-rollout, GRESO:** skip a prompt with probability 1 − p_e^z, where z is its run of consecutive zero-variance
    groups. p_e starts at 0.5 with a floor of 0.05 and moves in steps of 0.01 toward a 25% target, split 1:2 easy:hard.
    It matched dynamic sampling with 2.1–3.35x fewer rollouts [V, math].
- **Why for us:** 57% zero-variance groups means fewer than half of the 128 completions teach anything [R].
- **Default:** the GRESO-style pre-rollout skip, because it saves the most generation time.

### L1. Learnability prioritization
- **What:** sample prompts in proportion to p(1−p).
  - LILO: draw a 4x candidate pool, roll out 8 attempts, keep the top-|B| by p̂(1−p̂). It reaches the baseline's best
    1.5–3.3x sooner, and its GRPO arm at 1.5B gains +1.6 pp [V].
  - Bae et al.: keeping 0.3 < p < 0.7 beat plain GRPO by +2–4 pp, while 0 < p < 1 helped less [V].
- **Our version [P]: keep a Beta(α, β) posterior on p per prompt.**
  - The prior comes from the difficulty predictor (§3).
  - Each group of 8 updates it (α += passes, β += fails, with an exponential forget factor so the policy can move).
  - Sample prompts by Thompson draws of p(1−p).
- **Why not LILO's 4x pool:** it costs 4x generation on GPUs, where rollouts dominate [V]. With G = 8, p̂ from a single
  step has an SE of about 0.18 at p = 0.5 [V], so a single-step band is noisy. A posterior that pools steps
  fixes both problems.
- **Theory note:** the gradient's dependence on p(1−p) is strongest without std normalization (Dr. GRPO, RLOO).
  GRPO's normalization removes it in theory but not in practice (r = 0.384) [V]. Prediction: L1 helps the Dr. GRPO
  arm more than the GRPO arm. Our arms test this directly [P].

### L2. Retire mastered prompts
- **What:** drop a prompt for the rest of training once its posterior mean is ≥ 0.9 over at least 2 visits. ScaleRL and
  Polaris drop prompts with historical pass rate ≥ 0.9 [V-reported; the exact rule was refuted 0-3, so treat it as
  reported, not confirmed].
- **Why:** with a pool of about 44k ([01](01-data.md)), mastered prompts crowd out the learnable ones.

### L3. Zero-pass prompts: compiler-prefix guidance
- **What:** for prompts with posterior mean below 0.02 after at least 16 attempts, build a **guided** copy. The prompt
  is the x86 plus the first fraction f of the **compiler's reference target assembly**, with an instruction to
  continue. The model generates the rest, and the full program is graded.
  - **Choose f by POPE's rule** [V]: try evenly spaced f values and keep the shortest that gives any success. If none
    succeeds, use a random f below 0.25.
  - **Alternative, QuestA's schedule:** f = 0.5 for the first ~5% of steps, then 0.25 [V].
  - **Cut at boundaries:** cut only at function boundaries, or at least at basic-block boundaries, so a prefix never
    splits an instruction sequence the model has to understand as a whole [P].
  - **Mix and eval:** train guided and unguided copies 1:1 [V], and always evaluate unguided.
- **Batch placement:** in POPE, mixing unguided easy prompts into a hard-only phase badly hurt progress on the hard set
  ("ray interference"), and guidance prevented it [V, single source, low confidence]. So guided prompts go in their own
  phase, or in batches where they hold a fixed share.
- **Evidence size:**
  - R3: +4.1 over RL [V].
  - QuestA: AIME24 61.8 → 72.5, but 69.5 for plain RL on the same hard-filtered data [V].
  - POPE: held-out AIME25 +3.5 pp [V].
  - All three are math, single runs. Expect a few pp, not a breakthrough.
- **Prefixes lengthen prompts that are already long** (BringUp O2 prompts have a median of 5.7k tokens and p90 of
  23k [R]). Splice mode (§1.L4.a) is the alternative for whole programs.

### L4. Structural stages (where static difficulty belongs)
Each axis is a ladder of task categories. We **shift the mixture** between categories instead of switching hard
from one to the next, because strict ordering failed to beat random in 11 of 18 cells [V]. Use SEC's bandit, whose arm
reward is the batch mean |advantage|, which tracks p(1−p), with an EMA α of 0.2–0.5 and a softmax temperature τ of
0.2–1.0 [V]. It needs no extra rollouts.

| Axis | Rungs | Why it may help | Evidence |
|---|---|---|---|
| **a. Unit** | function (MBPP-C) → contest whole programs (stdio) → contest programs in splice mode → long whole programs (BringUp-like) | Run 1 improved functions but not whole programs [R]. Contest programs are a middle rung, and splice mode gives dense credit on whole programs | [R] + [P] |
| **b. Optimization** | O0 → O2 | GG: 99.4% at O0 vs 45.1% at O2 (ARMv8 HumanEval) [R paper]. O0 tasks give learnable signal on the same programs | [R] + [P] |
| **c. Length / context** | completion cap 4k → 8k → 16k | 17% of GG's BringUp failures were context truncation [R paper]. Run 1's cap of 4096 covers all but 11 of 952 train references [R]. DeepScaleR's 8k→16k→24k schedule | [R]; DeepScaleR [L, unverified] |
| **d. Target** | ARMv8 macOS → AArch64 Linux → RISC-V64 | GG: ARMv8 is closest to x86 (chrF 40.2) and RISC-V the furthest (21.4), and accuracy follows [R paper] | [R] + [P] |

## 2. SFT curriculum
- **No verified evidence** for curricula in SFT on code or translation. The Platanios competence curriculum and the
  length curricula did not pass verification in this run.
- **Proposal [P]:**
  - Batch SFT by length bucket for throughput only.
  - Run **one** curriculum ablation in the SFT screen, on a subset: random mixture vs O0-first-then-mixed.
  - Do not build a deeper SFT curriculum unless that ablation shows ≥ 2x the screen's seed spread.
- **What does matter for SFT, per the evidence:**
  - context length (GG: 4k → 16k gave +10 pp, about 16 HumanEval problems [R paper]);
  - unique data scale [V];
  - choosing the RL start by validation loss and pass@large-k, not by SFT pass@1 [V] ([06](06-schedule.md)).

## 3. Difficulty predictors (cheap, run first)
**Experiment CU-0 (no training, minutes to hours on the eval lanes) [P]:**
- **Ground truth:** base GG's pass rate over n = 16 samples per prompt on every RL pool and eval set. This is the
  Phase-0 census in experiments.md.
- **Features to test:**

  | Family | Features |
  |---|---|
  | static, x86 | instruction count, basic blocks, loops and their nesting depth, number of call sites and functions, SIMD/FP instructions, memory-operand share, literal pools, jump tables |
  | source | cyclomatic complexity, LOC |
  | task | optimization level, target, number of tests |
  | metadata | Codeforces rating, AOJ volume |
  | model-based | GG's mean token NLL on the reference target asm (one forward pass, no sampling) and on the x86 prompt |

- **Report:**
  - Spearman correlation with the pass rate;
  - AUROC for "pass rate = 0" and for "pass rate ≥ 0.9", each with a bootstrap CI;
  - a small gradient-boosted model's held-out R².
- **Use:** the best predictor sets the Beta prior in L1, so new prompts start near their likely pass rate. Its quality
  also decides which static axis is worth a stage in L4.
- **Expectation:** the model-based NLL probably predicts best, and static features probably predict weakly. AdaRFT
  scored difficulty with a reference model's pass rate over 128 samples [V], and cites pass@40–64 as enough to keep the
  ordering. This expectation is untested on code.

## 4. Measurement rules for curriculum arms
- Plot every curriculum arm against **generated tokens and GPU-hours**, not steps. The layers change how many rollouts a
  step costs.
- Fit ScaleRL's sigmoid R − R0 = (A − R0) / (1 + (C_mid/C)^B) per arm after at least one epoch over the pool [V]. Treat a
  difference in A below 0.02 as noise [V].
- Report the share of zero-variance groups, the pool composition over time and the number of retired prompts per arm.
- Evaluate unguided, on held-out problems only ([05](05-statistics.md)).

## 5. What we expect, honestly
| Layer | Expected effect | Confidence |
|---|---|---|
| L0 | up to about 2x useful signal per rollout (57% zero-variance today); 2–3x fewer rollouts at matched accuracy | high on the mechanism, medium on the size (math evidence) |
| L1 | +1–4 pp, faster convergence | medium (math only, single seeds) |
| L2 | efficiency only | low–medium |
| L3 | turns some zero-pass prompts learnable; a few pp on hard held-out sets | medium (3 groups, math) |
| L4 | unknown; this is our own contribution to test | — |

## Sources
Verified claims: [evidence/03-curriculum.md](evidence/03-curriculum.md).
- LILO: arXiv 2502.12272; Bae et al. (online difficulty filtering): arXiv 2504.03380; GRESO: arXiv 2506.02177
- SEC: arXiv 2505.14970; AdaRFT: arXiv 2504.05520; ScaleRL: arXiv 2510.13786
- R3: arXiv 2402.05808; QuestA: arXiv 2507.13266; POPE: arXiv 2601.18779
- GG ablations (context, data, O0 vs O2, chrF by ISA): arXiv 2506.14606 (repo PDF)
