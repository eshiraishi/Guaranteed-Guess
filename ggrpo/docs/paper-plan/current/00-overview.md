# Paper plan: what a test reward adds beyond imitation (x86 → ARMv8, GRPO vs controls)

**Status (Nov 10, 2026): the plan we can run now.** It uses only current resources and data we may use (Enzo, Nov 10:
"based solely on what we can now"). The Oct 24 design for ideal resources stays in the parent folder ([../00-overview.md](../00-overview.md)),
marked as not the current plan.

## Files
| File | Content |
|---|---|
| [01-data.md](01-data.md) | sources, splits, decontamination, the RL pool, eval sets |
| [02-harness.md](02-harness.md) | the training stack and the reward path as they run today |
| [05-statistics.md](05-statistics.md) | endpoints, tests, power on our sets, seeds, pre-registration |
| [06-schedule.md](06-schedule.md) | dated schedule on our GPUs, with gates |
| [../evidence/](../evidence/) | the Oct 24 deep-research outputs (verified claims with quotes) |

Tags: **[V]** verified in `evidence/`; **[R]** measured in this repo; **[L]** literature not verified; **[P]** our proposal.

## 1. The study
**Paper:** one, for ICML 2027 (deadline assumed late January 2027; check). Draft: Obsidian `AI/Pending/GG RLVR paper
draft (Nov 10)`.

**Question:** SFT imitates one reference per program (the compiler's output); a test reward accepts any correct output
and lets the model learn from its own samples. What does the test reward add beyond imitation?

**Hypotheses** (operational form in [05](05-statistics.md)):
- H1: GRPO beats its SFT start on E1 and E2.
- H2a: GRPO beats SFT on the same RL problems' compiler references.
- H2b: GRPO vs rejection-sampling fine-tuning (RFT) at the same sample budget (two-sided).
- H3: GRPO beats GRPO with permuted rewards, and the permuted arm stays within ±Δ_eq of the SFT start.

**Arms**, all from the same SFT checkpoint and the same RL pool:

| Arm | Trains on | Tests |
|---|---|---|
| SFT start | 200k pairs, GG recipe | reference |
| SFT on RL references | the pool's compiler references | H2a |
| RFT | the SFT model's passing samples from the census (16 per task = GRPO's budget) | H2b |
| GRPO | test reward | H1–H3 |
| GRPO, permuted reward | each group's real rewards shuffled within the group | H3 |

## 2. Resources (the limits this plan is built on)
| Resource | Use | Measured cost |
|---|---|---|
| totoro GPU0, RTX PRO 6000 Blackwell (96 GB) | training only | SFT 200k × 2 epochs: 2,084 updates × 46 s ≈ 27 h [R]; GRPO 16 prompts × 8, 16k cap: ~17–22 min/step (Nov 9 pilot, stub reward) [R]; 32 × 8: ~55 min/step [R] |
| totoro GPU1 and ponyo, 4070 Ti SUPER (16 GB) | evals, census | beam-8: ~13 h per ~190 CodeNet tasks per GPU [R]; vLLM sampling: 6,400 completions in ~1.6 h (census16) [R] |
| cluster RTX 5000 Ada (32 GB) | BringUp beam-8 | [R] Oct 23 |
| Mac, Tart VM `gg-ventura` (macOS 13.6, Xcode 14.3.1, 8 workers) | every reward and score | worst batch ~86 s [R]; must stay awake and online during RL |

## 3. Data (only sources we may use)
GG's released ARMv8 -O2 pairs; Project CodeNet accepted C (CDLA-Permissive-2.0); MBPP-C; HumanEval-C and BringUp
for evaluation; AOJ official tests for evaluation only, never redistributed. No Codeforces tests, no AtCoder tests
([01](01-data.md)).

## 4. Locked decisions
| Date | Decision |
|---|---|
| Oct 23 | real tests only; SFT data control |
| Oct 24 | co-primary endpoints E1 (beam-8) and E2 (avg@16) |
| Nov 6–7 | split by problem, 60/20/20; test frozen by hash, scored once |
| Nov 8 | decontamination protocol (planted-copy recall) |
| Nov 10 | base Qwen2.5-Coder-1.5B-Instruct (rev 2e1fd397); GG's released SFT recipe, cosine; RL starts from the final SFT checkpoint |
| Nov 10 | RL problems out of SFT (D5, option A); fp16 for RL; permuted-reward control |
| Nov 10 | ICML framing; H1 as an intersection–union test; RFT arm; AOJ full tests as robustness endpoint |
| Nov 10 | plan limited to current resources and safe data |

## 5. Open gates
| Gate | Blocks | Where |
|---|---|---|
| B-gate: SFT ≥ 72/164 HumanEval-C beam-8 | RL | [06](06-schedule.md) |
| C-0: GRPO config, from evidence plus a short pilot (step time, stability) | RL runs | [06](06-schedule.md) |
| Census → RL pool, steps, completion cap | RL runs | [06](06-schedule.md) |
| `prereg.md` committed | any confirmatory run | [05](05-statistics.md) §6 |
| CodeNet test length rule (27 of 370 tasks need > 32,768 tokens) | test scoring | [05](05-statistics.md) |
| Audit spot-check by Enzo | publishing audit precision | Obsidian `Audit spot-check guide` |
