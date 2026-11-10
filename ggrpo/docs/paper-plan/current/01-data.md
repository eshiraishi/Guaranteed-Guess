# 01. Data

**Status (Nov 10, 2026): built, except the census-filtered RL pool and the AOJ full tests.** Only sources we may use.
Tags as in [00](00-overview.md).

## 1. Sources
| Source | Use | Size | Terms | Tag |
|---|---|---|---|---|
| GG SFT (`ahmedheakl/gg-armv8-O2-instruct`) | SFT | 1,309,052 pairs (AnghaBench 1,037,405; Stack 271,647) | released by the GG authors | [R] |
| Project CodeNet, accepted C (`iNeil77/CodeNet` 1e28cfa) | SFT (sft_only problems), RL pool, eval | 260,599 solutions built on our toolchain | CDLA-Permissive-2.0 | [R] |
| CodeNet statements (`Elfsong/codenet_metadata` b0e2643) | RL and eval tests: the statement's sample I/O | 2,067 problems with ≥ 2 distinct sample outputs | CDLA-Permissive-2.0 | [R] |
| MBPP-C (decompile-eval) | RL pool, dev | 849 train, 90 dev (13 HumanEval twins dropped) | CC-BY-4.0 | [R] |
| HumanEval-C, GG bench + the authors' tests | eval | 164 (152 without MBPP twins) | MIT | [R] |
| BringUp, -O2 | eval | 65 (53 clean) | Apache-2.0 | [R] |
| AOJ official tests (judgedat.u-aizu.ac.jp) | eval only (E8), never redistributed | val/test AOJ problems: 211 of 287 mapped | terms silent; crawl approved Nov 6, 1 request/s | [R] |

**Not used:** Codeforces tests (license), AtCoder tests (terms), synthetic or model-written tests.

## 2. Splits (Nov 7)
- CodeNet RL problems split 60/20/20 by problem group (sha256 with a fixed salt), so problems that share samples never
  cross splits. Test is frozen by content hash and scored once, at the end.
- One task per problem in val (413) and test (370). By judge: val 151 AOJ / 262 AtCoder; test 136 AOJ / 234 AtCoder.
- CodeNet sft_only problems (1,156) feed SFT only.

## 3. Decontamination and keeping RL problems out of SFT
- Protocol (Nov 8): asm containment, C containment and exact rules, with planted-copy recall checked in every run.
  Details: Obsidian `Decontamination protocol`.
- D5 (Nov 10, option A): RL tasks are decontamination items too, and their drops apply to SFT only. SFT also takes no
  CodeNet train problem, no MBPP-C task and no exact x86 copy of an RL task (24 rows).
- Cost: eval items drop 0.40% of GG and 5.5% of CodeNet documents; eval and RL items together drop 1.56% of GG and
  13.5% of CodeNet SFT documents (budgets 2% and 15%).

## 4. What each arm trains on
- **SFT:** `data/sft/train.parquet`, 1,087,411 rows after dedup, decontamination and a 16,384-token cap per side
  (CodeNet 46,778; AnghaBench 882,667; Stack 157,966). The SFT start uses a fixed 200k random subset (seed 20261110:
  CodeNet 8,662; AnghaBench 162,049; Stack 29,289), val 472.
- **RL pool before the census:** `data/rl/train.jsonl`, 7,278 tasks (CodeNet 6,492 from 1,017 problems, ≤ 10 solutions
  each; MBPP-C 786).
- **RL pool used by every arm [P]:** the tasks whose pass rate over 16 samples from the SFT model is strictly between 0
  and 1. Only those give GRPO a gradient; GG's model had a mixed group of 8 on 17% of CodeNet val tasks [R]. The same
  16 samples per task are RFT's training data.

## 5. Evaluation sets
| Set | Tasks | Tests | Role | Detectable effect, E1 [P] |
|---|---|---|---|---|
| CodeNet test | 370 problems | statement samples | **co-primary (E1, E2)** | ≈ 4.5 pp |
| CodeNet test, AOJ part | ≤ 136 problems | AOJ official tests | robustness (E8) | ≈ 7.5 pp |
| HumanEval-C | 164 / 152 | authors' asserts | comparability with GG | ≈ 6.8 pp |
| BringUp | 65 / 53 clean | upstream tests | whole programs | ≈ 11.9 pp (clean) |
| CodeNet val, MBPP-C dev | 413, 90 | samples, asserts | monitoring only | — |

Detectable effects use Miller's formula with HumanEval's measured discordance (ω² ≈ 0.095); recompute with each set's
own discordance ([05](05-statistics.md)).
