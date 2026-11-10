# 05. Statistical protocol

**Status (Nov 10, 2026): to be frozen in `prereg.md` before the first confirmatory run.** Tags as in [00](00-overview.md).
Verified claims: [../evidence/05-stats.md](../evidence/05-stats.md).

## 1. Endpoints
| Endpoint | Definition | Role |
|---|---|---|
| **E1: beam-8 accuracy** | the top output of beam search (8 beams, `do_sample=False`, repetition penalty 1.1, cap 32k − prompt) passes all tests; GG's protocol | co-primary, CodeNet test |
| **E2: avg@16** | mean pass rate over 16 samples per task at the training settings (T 1.0, top-p 1, top-k 0), averaged per task first | co-primary, CodeNet test |
| E3: pass@k, k ≤ 64 | unbiased estimator (Chen et al., 2021) from 64 samples per task; E2 uses the first 16 | secondary: sharpening vs new coverage |
| E5: HumanEval-C | E1 on 164 and on the 152 without MBPP twins | secondary: comparability with GG |
| E6: BringUp | E1 on 65 and on the clean 53 | secondary |
| E8: full tests | E1 and E2 rescored with AOJ's official tests on the AOJ test problems; false-pass rate = share of sample-test passes that fail them, per arm | secondary: does the gain survive tests the reward never saw |

Also reported, not tested: gain by output length; how far correct outputs differ from the compiler's; results by
similarity stratum (`ggrpo.strata`; clean is the headline when high beats clean by > 10 points).

## 2. Hypotheses and tests
| Hypothesis | Comparison | Endpoints | Test |
|---|---|---|---|
| H1 | GRPO > SFT start | E1 **and** E2 | one-sided, each at α = 0.05; holds only if both reject (intersection–union, no correction across the two) |
| H2a | GRPO > SFT on RL references | E1, E2 | one-sided |
| H2b | GRPO vs RFT | E1, E2 | two-sided |
| H3 | GRPO > permuted reward; permuted within ±Δ_eq of SFT start | E2 | one-sided superiority; TOST for equivalence |

- **Family:** H2a, H2b and H3-superiority form one family per endpoint, Holm-corrected. Raw and adjusted p side by side.
- **E1 (one binary outcome per task):** exact McNemar (sign) test on discordant pairs; Wilson interval per arm.
- **E2, E3 (samples per task):** average per task first, then a paired sign-flip permutation test over tasks.
- **Seeds:** average each task over seeds before the paired test; report every seed and the seed SD.
- **Δ_eq = 3 pp on E2 [P]**; its power is recomputed from the census's per-task variance before `prereg.md` is frozen.
  A null without equivalence is reported as "inconclusive below the detectable effect".

## 3. Power on our sets
Miller's formula, δ = 2.80·sqrt(ω²/n), with HumanEval's measured ω² ≈ 0.095 (E1):

| Set | n | Detectable effect |
|---|---|---|
| CodeNet test | 370 | ≈ 4.5 pp |
| CodeNet test, AOJ part | ≤ 136 | ≈ 7.5 pp |
| HumanEval-C | 164 | ≈ 6.8 pp |
| BringUp clean | 53 | ≈ 11.9 pp |

E2 averages 16 samples per task, so its detectable effect is smaller; recompute both with measured ω² and σ² before
freezing.

## 4. Seeds [P]
- **Default:** 3 seeds for every RL arm and every SFT-type control, if C-0 measures ≤ 3 days per GRPO run
  ([06](06-schedule.md)). The three GRPO seeds are also the noise floor.
- **Fallback, if runs are slower:** GRPO 3 seeds, each control 1 seed; a control whose E2 difference to GRPO is within
  two GRPO seed SDs gets two more seeds.
- The choice is fixed in `prereg.md` from the measured step time, before any confirmatory run.

## 5. Selection and fairness
- The reported checkpoint of every run is its final one, of a length fixed before the run. Val and dev monitor runs;
  they never pick checkpoints or arms. CodeNet test, HumanEval-C and BringUp never pick anything.
- All arms share the SFT start, the RL pool, decoding and the eval stack. RFT's training data are the census's 16 samples
  per task, GRPO's sample budget (8 samples × 2 passes). GRPO and the permuted arm run the same steps.

## 6. Pre-registration
Commit `docs/paper-plan/current/prereg.md` before the first confirmatory run, with: endpoints; hypotheses and tests; family and
Holm; Δ_eq; seed rule; eval sets with content hashes; the CodeNet test length rule; checkpoint rule; decoding settings;
the census filter and step count. Report any deviation in the paper.

## 7. Reporting
- Every arm that ran, including failed ones, with config and seed.
- Per arm and endpoint: accuracy with interval; Δ vs the reference with up/down counts; raw and Holm p.
- The detectable effect next to every null.
- Compute: GPU-hours, generated tokens, reward CPU-seconds.
- Per-task score files for every number.
