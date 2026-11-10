# 06. Schedule

**Status (Nov 10, 2026): dated plan on current hardware.** Times in GST. Costs are measured [R] or estimated [P].
GPU0 trains only; evals and the census run on the 4070s and the student lab ([02](02-harness.md) §4).

## 1. Timeline
| When (GST) | GPU0 (Blackwell) | Eval GPUs (4070s, lab) | Gate |
|---|---|---|---|
| to Nov 11 ~20:20 | SFT v2 (running since Nov 10 17:25; 46 s/update) [R] | SFT checkpoints: samples and dev, every 10% | — |
| Nov 11–12 | C-0 pilot: ~20 GRPO steps on the SFT model with the real reward (~7 h) | B-gate HumanEval-C; census: 16 samples × 7,278 RL tasks from the SFT model (~30 GPU-h on 4070s [P]) | B-gate; C-0 |
| Nov 13 | SFT on RL references and RFT, 3 seeds each (~1 h per run [P]) | score the census; fix the pool and the step count | `prereg.md` committed |
| Nov 13 – Dec 1 | GRPO seeds 0–2, permuted reward seeds 0–2 (~2.5 days each [P]) | val and dev monitoring of each run | — |
| Dec 1–12 | buffer: reruns, fallback seeds ([05](05-statistics.md) §4) | — | — |
| Dec 1–15 | — | final scoring, once: CodeNet test (E1, E2, E3, E8), HumanEval-C, BringUp, for the SFT start and every arm and seed (13 models) | CodeNet test length rule |
| Dec 15 – Jan | — | analysis, figures | — |
| late Jan 2027 | — | ICML 2027 submission (date to check) | — |

The Mac must stay awake and online from Nov 11 to Dec 1 (rewards) and during final scoring.

## 2. Gates
- **B-gate:** the final SFT checkpoint reaches ≥ 72/164 on HumanEval-C beam-8 (GG's model on our stack). If it is lower,
  debug data, template and tokenizer before RL.
- **C-0:** the config comes from evidence, not from the pilot, which is too short to rank configs: GRPO with a
  token-level loss, β = 0, LR 1e-6 (schedule: open), 16 prompts × 8, temperature 1.0, completion cap 16k (or 8k if the
  census shows no pass above 8k), truncated completions kept with reward 0, fp16. The pilot only measures the step time
  and checks for NaN, collapse and reward-server errors. 32 × 8 and 500 steps are out (~55 min/step at 32 × 8) [R].
- **Census → pool:** keep the RL tasks with 0 < pass rate < 1 over 16 samples. Steps = 2 passes over the pool at 16
  prompts per step. If the pool is so large that a run exceeds 3 days, cap the pool by a fixed random draw, before
  any run.
- **`prereg.md`:** endpoints, tests, Δ_eq, seed rule (from the measured step time), pool and step count, hashes
  ([05](05-statistics.md) §6).
- **CodeNet test length rule:** 27 of 370 test tasks need more than 32,768 tokens and cannot pass; decide whether they
  count as failures or are excluded, before scoring.

## 3. Costs behind the timeline
| Item | Unit cost | Count | Total |
|---|---|---|---|
| GRPO run | ~20 min/step [R] × ~190 steps [P: 2 passes over ~1,500 tasks] | 6 | ~15 days GPU0 |
| SFT-type controls | ~1 h [P] | 6 | < 1 day GPU0 |
| Census | 116k completions at ~4k/h per 4070 [R census16] | 1 | ~30 GPU-h |
| E1 on CodeNet test | ~13 h per 190 tasks per 4070 [R] | 13 models | ~340 GPU-h on 4070s, less on the lab's 32 GB cards [P: benchmark] |
| E2/E3 on CodeNet test | 64 × 370 samples, ~6 h per 4070 [P] | 13 models | ~80 GPU-h |
| Scoring in the VM | ~1.5 completions/s [P from run 1] | ~310k completions | ~2.5 days of VM time |

## 4. Not in this plan
Curriculum, synthetic tests, multi-target or RISC-V, CoT, efficiency reward, async training, GG-XL, Codeforces and
AtCoder data. They need resources or data we do not have; the Oct 24 design in the parent folder describes them.
