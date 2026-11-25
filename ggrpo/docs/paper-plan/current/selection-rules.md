# Development selection and stop rules (validation only)

**Fixed Nov 21, 2026 (Enzo: "Yes, all four", adopting the v7 review simulation's changes 1–4), before any Phase 2 run
and before this session read any Phase 1 E2 of an RL checkpoint.** The git commit time of this file is the timestamp.
It supersedes the selection rule in `ops/lanes/blackwell_queue.sh` (held Nov 20) and the Phase 2 rule of the action plan
(Obsidian, "GG RLVR action plan after the first runs"). Test, HumanEval-C and BringUp are not used for anything here.
**Amended Nov 22, 2026 (Enzo):** §1 and §4 follow `prereg.md`'s amended pass, because the differential suite failed its
AOJ check (Nov 21). Any setting chosen under the old pass is chosen again under this one. The GRPO E1/E2 values seen
under AOJ's official tests (Nov 21–22) were on validation and select nothing.

## 1. Pass and metric
- **Pass:** the sample tests (`prereg.md` §4, amended Nov 22); selection uses it on all 383 val tasks. **Official
  pass:** sample tests and the official tests, on the 107 AOJ val tasks and the 359 Codeforces val tasks, pooled
  (Codeforces added Nov 22, before any Codeforces val number was read); reported next to every candidate, used by the stop rule (§4). The differential suite is not used.
- **E2:** per task, the fraction of 16 samples that pass; averaged over the 383 CodeNet val tasks. pass@k (k = 1..16)
  comes from the same files with the unbiased estimator; it is descriptive.

## 2. Same tuning budget for every trained arm
| Arm | LR grid | Horizon grid (passes over its data) |
|---|---|---|
| GRPO, online RFT | 5e-7, 1e-6, 2e-6 | 1, 2 (checkpoints 184 and 368; the LR is constant after warmup, so a checkpoint equals a shorter run) |
| SFT on RL references, SFT on pool references | 5e-6, 1e-5, 2e-5 | 1, 2 (separate 1- and 2-epoch runs; under the cosine schedule a mid-run checkpoint is not a shorter run) |

- **Seeds:** one per setting (seed 0). Every setting within 1 pp of its arm's best then gets seed 1; a setting's score is
  the mean over its seeds. Every arm follows the same rule (GRPO at 1e-6 already has seed 1).
- **Grid edge:** if an arm's best LR is at the edge of its grid, one more LR one step outward (×2 or ÷2) is run, once,
  with the same seed rule.
- **Ties** within 0.5 pp go to the anchor: LR 1e-6 and 2 passes for the online arms, LR 2e-5 and 2 epochs (GG's recipe)
  for the SFT arms.
- The permuted-reward control has no grid; it inherits GRPO's chosen setting.

## 3. Sampling temperature
After each arm's (LR, horizon) is chosen at T = 1.0, the chosen model of every arm and the SFT start get E2 at
T ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} (T = 0 is greedy, one sample per task). Each model's temperature is its argmax; ties
within 0.5 pp go to the higher temperature. This makes "GRPO only sharpens the start" testable: the start is compared at
its best temperature, not at the training temperature.

## 4. Development gate for H1 (stop rule)
GRPO at its chosen (LR, horizon, temperature) against the SFT start at its chosen temperature, on E2 under the
pass and again on the 107 AOJ and 359 Codeforces val tasks, pooled, under the official pass, each with a task-paired 95% bootstrap interval of the
difference. If either lower bound is at or below 0, the
GRPO path stops for ICML 2027 and the results are reported as they are. The gate is a go/no-go on validation; because
the same validation tasks chose the settings, its estimate is optimistic and is not a claim. The claims come from the
confirmatory runs (fresh seeds) on test, where E1 (beam 8) stays the primary endpoint and H1 also needs the official pass (`prereg.md` §5).
