# 06. Experiment schedule: phases, values, sweeps and decision rules

**Status (Oct 24, 2026): plan, designed for ideal resources first and then scaled to ours (§9).**
Tags: [V] verified, [R] repo, [L] unverified literature, [P] proposal. Every value carries a reason or is marked as a
sweep. "Gate" means a written decision rule checked before the next phase starts.

## 0. Principles
1. **One switch per comparison.** Each arm differs from its parent in one setting. Data, order, steps, sampling, engine
   and eval stack are shared [R rule].
2. **Screen cheaply, confirm expensively.** Screening uses 1 seed and at least one epoch over the pool before any ranking,
   because early rankings invert [V]. Confirmation uses 3–5 seeds ([05](05-statistics.md) §5).
3. **Select on dev; report on held-out** (GG-XL, HumanEval-C, BringUp) [R].
4. **Plot against compute** (generated tokens, GPU-hours), not steps [V].
5. **Pre-register Phase E** ([05](05-statistics.md) §7).

## 1. Phase map
| Phase | Track | Content | Depends on | Gate to pass |
|---|---|---|---|---|
| **A** | foundations | harness v2, data build, GG-XL, census, predictors, MBPP-C test audit | — | A-gate |
| **B** | SFT redo | base screen, recipe sweep, full multi-target SFT, SFT controls | A | B-gate |
| **C** | RL recipe | regime, loss family, batch and G, precision | B | C-gate |
| **D** | data and curriculum | real-test scale, curriculum layers, guidance, splice credit, unit/length/target stages | C (D-1 can run on GG's checkpoint in parallel) | D-gate |
| **S** | synthetic arm (P2) | pipeline validity, adequacy curves, RL utility, iteration | A (validity), C (RL utility) | S-gate |
| **X** | extensions | CoT arms, efficiency reward, test-time scaling | C | per arm |
| **E** | confirmatory | pre-registered final runs, all endpoints, all targets | B–D, S | — |

## 2. Phase A: foundations (mostly CPU and the eval lanes)
| ID | Work | Output | Reason |
|---|---|---|---|
| A-1 | Harness v2: router, cache, telemetry, stdio mode, splice mode, static checks, AArch64-Linux and RISC-V backends ([02](02-harness.md)) | reference preflight: every reference passes and every stub fails, per target | rewards are the bottleneck [R]; new task types need new modes |
| A-2 | Data build: Codeforces (complete + partial), AOJ crawl (after terms are cleared), multi-solution with dedup cap k = 10; build per target; contamination scans ([01](01-data.md)) | task tables with `in_sft`, exclusion lists, split by problem | about 44k real-test prompts vs 858 [R] |
| A-3 | GG-XL: ≥ 1,000 held-out problems, stratified, frozen with a content hash | eval set + MDE recomputed | detectable effect drops from ~7 to ~3 pp [V] |
| A-4 | Census: base GG at n = 16 (training sampling) on every pool and eval set; beam-8 on evals | per-prompt pass rates, zero-variance share at G = 8/16, verdict mix, truncation | sets the curriculum priors and the RL pool [R plan] |
| A-5 | CU-0, difficulty predictors ([03](03-curriculum.md) §3) | Spearman/AUROC per feature | decides L1 priors and L4 stages |
| A-6 | S-3 early: FPR of MBPP-C's own asserts against an audit suite ([04](04-synthetic-tests.md)) | how many current "passes" are wrong | if it is high, our reward has false positives (P0) |
| A-7 | Step-time split and cache hit rate on the current stack | generation / reward / train seconds per step | chooses async vs reward scale-out [V] |

**A-gate:**
- Every reference passes on every target, or the task is dropped.
- Stubs pass 0 tasks.
- GG-XL is frozen.
- The census is done.
- The MBPP-C assert FPR is known. If it is above 5%, the stronger tests go into the reward before Phase C.

## 3. Phase B: SFT redo ("why redo it": [00](00-overview.md) §3)
### B-0. Base-model screen
- **Candidates** (about 1–2B, all open):
  - Qwen2.5-Coder-1.5B **Instruct** (GG's base, the replication anchor);
  - Qwen2.5-Coder-1.5B **Base**;
  - Qwen3-1.7B-Base;
  - DeepSeek-Coder-1.3B. GG already reports this one far behind: 10.3% vs 45.1% on HumanEval O2 [R paper].
    It serves as a sanity anchor and is dropped if the screen agrees;
  - one or two newer 1–2B code models available at screen time [L].
- **Cheap measurements before any training:**
  - tokenizer fertility on our asm corpus, in tokens per instruction (GG Table 2 gives Qwen 3.62 vs DeepSeek 4.28 on
    ARMv8 [R]);
  - native context length;
  - throughput.
- **Training screen:** identical SFT on a fixed **200k-pair** multi-target subset. Use GG's recipe (LR 2e-5 cosine, warmup
  0.1, 2 epochs, output-only loss) at batch 192.
- **Rank by:**
  - validation loss on a fixed held-out SFT set;
  - pass@64 on dev with tests, at training sampling.

  These predict post-RL results better than SFT pass@1 [V]. Also report beam-8.
- **Keep the top two.** Qwen2.5-Coder-1.5B-Instruct is always kept as one of them, for comparability with GG.

### B-1. SFT recipe sweep, on the best base and the 200k subset (1 seed each)
| Knob | Values | Reason |
|---|---|---|
| LR | 1e-5, **2e-5**, 5e-5 | GG and LLM4Decompile both used 2e-5 [V]; the best LR depends on the model family [V] |
| Effective batch | **192**, 1,024 | GG used 192 [R]; LLM4Decompile used 2,048 [V]; larger batches win only late [V] |
| Epochs | 1, **2**, 3 (read from one 3-epoch run's checkpoints) | no verified assembly evidence for 2 [V refuted] |
| Context (train) | 16k, **32k** if the base supports it | GG: 4k → 16k gave +10 pp; 17% of BringUp failures were truncation [R paper] |
| Tokenizer extension | off, **on** (GG's opcode and register tokens) | GG: +1.04 pp [R paper]; it must be re-tested on a new base |
| Prompt format | target named in the prompt (multi-target) | needed for one multi-target model |

Defaults are in bold. Rank by validation loss at the **end** of each run, never early [V].

### B-2. Full SFT
- **Data:** GG's 1.31M programs plus new unique programs ([01](01-data.md) §2), rebuilt for 6 targets: ARMv8-macOS,
  AArch64-Linux and RISC-V64, each at O0 and O2.
  - One target is drawn per program per epoch, so tokens per epoch stay at GG's level.
  - RL and eval problems are excluded.
- **Checkpoints** every 0.25 epoch. **Choose the RL start** at the validation-loss plateau, the "stable regime" [V].
  Report pass@64 at each checkpoint.
- **Two bases × 1 seed** for the main runs, plus 2 more seeds of the primary base for the SFT noise floor.

### B-3. SFT controls and ablations, on the primary base
| ID | Change | Question |
|---|---|---|
| B-3a | per-target models vs one multi-target model | does multi-target help or hurt each target? |
| B-3b | data scale 10% / 30% / 100% | the scaling figure; does more unique data raise the post-RL ceiling here too [V math]? |
| B-3c | RL-pool problems included in SFT | the data-control question: does RL beat SFT on the same problems? |
| B-3d | O2-only vs O0+O2 | does O0 help O2? |

**B-gate:** each checked against GG's released models on the same eval stack:
- the primary base's SFT matches or beats GG on HumanEval-C beam-8 at ARMv8 O2 (≥ 72/164);
- RISC-V64 and O0 are within the paper's reported range.

If the redo is clearly below GG, debug before RL: data, chat template, tokenizer, RoPE settings.

## 4. Phase C: RL recipe, on the B-2 checkpoint
**Control recipe "RL-v1" (changes from run 1 in bold) [V-backed defaults]:**

| Setting | Value | Reason |
|---|---|---|
| Algorithm | GRPO with token-level loss (TRL `dapo`), β = 0 | minimal recipes match heavier ones at 1.5–8B (JustRL, Lite PPO) [V]; it is our existing control [R] |
| LR | **1e-6 constant**, 10-step warmup | JustRL kept 1e-6 constant over 4k+ steps [V]; run 1 decayed to 0, an average of about 5e-7 [R] |
| G | 8 | used by JustRL, Lite PPO and the 3B code run [V] |
| Prompts per step | **64** (512 completions) | run 1's 128 completions is 4x below the smallest batch tested; larger batches raised the asymptote in ScaleRL [V, grade C] |
| Steps | **≥ 500**; extend to 1–2k while dev rises | published runs use 400–4,380 steps [V]; 106 is too few to read [V] |
| ε (low / high) | 0.2 / **0.28** (active only once rollouts are off-policy) | clip-higher stabilizes entropy on aligned starts [V] |
| Logits | **FP32 LM head**, trainer and vLLM | ScaleRL: 0.52 → 0.61 asymptote with DAPO-style losses [V, split vote] |
| Temperature | 1.0; check entropy. If adjusted entropy is far above 0.3, add a T = 0.85 arm | JustRL/Lite PPO use 1.0; AceReason's heuristic [V, grade C] |
| Length | completion cap **8k** (from the census: cover ≥ 99% of references), truncated completions masked; no overlong penalty | an overlong penalty lowered JustRL's plateau and collapsed entropy [V, grade C] |
| Reward | binary pass on the full or capped real tests | the control |
| Batch filter | **effective batch (L0)** | 57% zero-variance groups in run 1 [R]; ScaleRL [V] |
| Data | real-test pool from A-2 (function + stdio), split by problem | [01](01-data.md) |
| Precision | fp32 master weights, bf16 autocast | as in run 1 and GG's SFT [R] |

**C-1. Regime (1 seed each, at least one epoch over the pool, ScaleRL fit) [P]:**
- LR ∈ {5e-7, **1e-6**, 2e-6, 4e-6}. Pick the highest that does not collapse entropy or reward and whose fitted A is
  within 0.02 of the best.
- Steps: extend the best LR until dev avg@16 is flat for 2 evaluation points.

**C-2. Batch and G, compute-matched at equal generated tokens [P]:**
- 16×8, **64×8**, 32×16 and 128×8. This tests ScaleRL's batch claim at 1.5B on code [V open question].
- G = 16 wins only if the census shows that it lowers the zero-variance share enough to pay for itself.

**C-3. Loss family, on-policy (1 seed, then 3 for finalists):**

| Arm | Change |
|---|---|
| GRPO (DeepSeekMath) | `loss_type grpo`, β 0.04 |
| Dr. GRPO | `dr_grpo`, `scale_rewards none` (≈ RLOO) |
| Lite PPO | group-mean / batch-std advantages, token-level loss [V] |

These are the paper's named baselines [R decision, Oct 23].

**C-4. Off-policy, only after async is in place (A-7, [02](02-harness.md)):**
- staleness η ∈ {1, 2, 4};
- losses: DAPO-token vs CISPO vs GSPO. CISPO and GSPO raise the asymptote over DAPO in ScaleRL [V].

**C-gate:**
- RL-v1 at the fixed regime beats its SFT start on dev avg@16 by more than twice the dev detectable effect, **or** shows a
  rising fitted A.
- The 3-seed noise floor ([05](05-statistics.md) §5) is measured.

## 5. Phase D: data scale and curriculum, on RL-v1
| ID | Arm | Parent | Measures | Evidence |
|---|---|---|---|---|
| D-1 | MBPP-C only vs + contest real-test pool (k = 1) vs k = 10 vs all solutions | RL-v1 | does real-test scale help; does solution diversity help | [R] counts; [V] unique-prompt scaling (math) |
| D-2 | dedup cap k ∈ {1, 3, 10} at an equal number of prompts | D-1 best | diversity vs more problems | [P] |
| D-3 | L0 (control) vs **L0 + GRESO pre-skip** | RL-v1 | rollouts saved at matched accuracy | GRESO 2.1–3.35x [V] |
| D-4 | + L1 learnability sampling (Beta posterior, predictor prior) | D-3 | +pp and efficiency | +1–4 pp [V math] |
| D-5 | + L2 retire p ≥ 0.9 | D-4 | efficiency | [V-reported] |
| D-6 | + L3 compiler-prefix guidance on zero-pass prompts (POPE rule, 1:1 guided/unguided, own phase) | D-5 | unguided pass on the hard stratum of GG-XL; BringUp | R3/QuestA/POPE [V] |
| D-7 | reward: binary vs **splice credit** (whole programs) vs assert fraction (functions) vs CodeRL ladder | D-5 | dense credit for whole programs | [P]; pass-rate reward mixed [R experiments.md] |
| D-8 | stages: unit (function → stdio → splice → long), O0 → O2, cap 4k → 8k → 16k, target order; SEC bandit vs fixed mixture | D-5 | do structural stages beat a fixed mixture | SEC [V]; ordered often < random [V] |
| D-9 | multi-target RL (one policy for all targets) vs per-target RL | best of D | transfer across ISAs | [P] |

**D-gate:**
- Each layer is kept if it improves dev avg@16 by more than the noise floor at **matched compute**, or if it matches
  accuracy with at least 1.5x fewer generated tokens.
- Otherwise it is dropped and reported as a null with its detectable effect.

## 6. Track S: synthetic arm (P2) ([04](04-synthetic-tests.md))
| ID | Work | Compute | Gate |
|---|---|---|---|
| S-0 | yield funnel on Jotai/AnghaBench, ExeBench C, Decompile-Bench C | CPU | — |
| S-1 | adequacy curves: FPR vs the number of inputs per generator on the calibration set (MBPP-C, HumanEval-C held out, DecompileBench) | CPU + sampling on the eval lanes | the k at which the 95% upper bound on FPR is ≤ 2% |
| S-2 | generator and gate ablation | CPU | FNR ≤ 0.5% |
| S-3 | real-test audit (done early as A-6) | CPU | — |
| S-4 | RL: R-only vs S-only vs R+S, compute-matched, 3 seeds | GPU | the pre-registered pool gate (FPR bound ≤ 2%) holds |
| S-5 | pool size 1k / 10k / 100k | GPU | — |
| S-6 | iteration rounds 0–3, FPR on held-out later-checkpoint negatives | GPU + CPU | FPR non-increasing |
| S-7 | hacking monitor: the suite-vs-audit gap over training | logging | gap ≤ 2 pp |

## 7. Track X: extensions (each a separate arm on RL-v1 or the D-best recipe)
| ID | Arm | Design | Reason / evidence |
|---|---|---|---|
| X-1 | **CoT, free-form** | the prompt allows `<think>…</think>` before the asm; RL only (no teacher traces exist); the length cap is raised and compute-matched | the decision is to study CoT as an arm; no verified evidence for translation; without teachers it is R1-Zero style [P] |
| X-2 | **CoT, tool-derived scaffold** | SFT then RL on a deterministic scaffold computed from the x86 by a disassembler or analyzer: CFG, a basic-block list, a stack-slot and register map, then the asm | GG suggests CFG/dataflow representations for O2 failures [R paper]; this uses no teacher model [P] |
| X-3 | **Efficiency reward** | r = 1[pass]·(1 + λ·clip(log(cost_ref/cost), −0.5, 0.5)), with a deterministic instruction-count cost; λ ∈ {0, 0.1, 0.3} | GG's systems claim; it is ablatable by construction (λ = 0 is the control) [P]; feasible on Linux targets first ([02](02-harness.md) §6) |
| X-4 | **Test-time scaling** | best-of-n with tests (E4) and repair turns from assembler/linker/test feedback, for n ∈ {1, 2, 4, 8, 16} at equal generated tokens | GG frames tests as the guarantee; no verified prior evidence for assembly [V gap] |

Each X arm is 1 seed at screening and 3 seeds if promoted. CoT arms are compared at equal **generated tokens** per task.

## 8. Phase E: confirmatory (pre-registered)
- **Arms:**
  1. GG (released);
  2. SFT-redo (each base);
  3. SFT data-control (B-3c);
  4. RFT on the SFT-redo, with the same tuning budget;
  5. GRPO (TRL default) and the named baselines (DeepSeekMath GRPO, Dr. GRPO) at the fixed regime;
  6. **the final recipe** (RL-v1 + the D layers that passed + the X arms that were promoted);
  7. the synthetic arm (S-4 best).
- **Seeds:** 5 for the final recipe and its main comparators; 3 for the others ([05](05-statistics.md) §5).
- **Endpoints:**
  - E1 and E2 on GG-XL (co-primary);
  - E3–E7, HumanEval-C (164/152), BringUp (65/53), DecompileBench robustness;
  - per target and optimization level.
- **Analyses:** exactly those in `prereg.md`.

## 9. Compute
### 9.1 Ideal plan (rough estimates) [P]
| Item | Unit cost | Count | Total |
|---|---|---|---|
| Full SFT, GG-size data at 16k context | GG: 240,833 s on 8×A100 ≈ 535 A100-h [R]; about half that on H100 [L] | 2 bases × 1 + 2 extra seeds | ~1,000–1,500 H100-h |
| SFT screen and sweep at 200k pairs | ~1/6 of a full run each | ~12 runs | ~600 H100-h |
| SFT ablations B-3 | ~4 subset-size or full-size runs | | ~800 H100-h |
| RL screening run (500–1,000 steps, 512 completions/step, async) | ~1–2 min/step on 8×H100 [P] | ~30 arms | ~3,000 H100-h |
| RL confirmatory (5 or 3 seeds) | | ~40 runs | ~4,000 H100-h |
| Synthetic pipeline | CPU: fuzzing and symbolic, ~10⁵–10⁶ core-h; GPU sampling for negatives | | ~200 H100-h + CPU |
| Evals (GG-XL × 16 samples + beam-8 on every checkpoint) | vLLM; beam-8 on HF | | ~1,000 GPU-h on smaller cards |
| **Total** | | | **~10k H100-h**, CPU fleet of ~128–256 cores, 4–8 Apple Silicon hosts |

### 9.2 Scaled to current hardware (1× RTX PRO 6000 Blackwell for training; 4070 Ti SUPERs and the cluster for evals) [P]
- **A:** fully feasible; CPU and eval lanes.
- **B:**
  - The B-0 screen at 200k pairs: one Blackwell run takes roughly a day (unmeasured).
  - **One** full SFT (the primary base): roughly 2–3 weeks.
  - B-1 shrinks to the LR × context cells.
  - B-3 is cut to B-3c (data control) at subset scale.
- **C:** RL-v1 at 64×8 and 500 steps takes about 4–5 days per run at today's ~200 s per 128 completions. It needs the
  reward fix (A-1/A-7) first; without it, cut to 32×8.
  - C-1 tests 2 LRs.
  - C-2 is skipped (keep 64×8).
  - C-3 keeps the named baselines.
- **D:** D-1, D-3/D-4 and D-6 only.
- **S:** S-0 to S-3 (CPU), and S-4 with 1 seed.
- **E:** 3 seeds for the final recipe and its 2 main comparators.
- **What to ask the authors for, in priority order:**
  1. one 8×H100/H200 node for about 6–8 weeks;
  2. 2–4 Apple Silicon hosts for macOS rewards;
  3. a 128-core Linux box (AArch64 if possible) for rewards and the synthetic pipeline.

  With (1) alone, the whole of B and C becomes feasible.

## 10. Order of work (critical path)
1. A-1, A-6, A-7 (harness, assert audit, step-time split) → A-2, A-3 (data, GG-XL) → A-4, A-5 (census, predictors).
   In parallel: S-0 to S-2 on CPU.
2. B-0 → B-1 → B-2. In parallel: D-1/D-3 on **GG's released checkpoint**. These preliminary results use the existing
   model, so the data and curriculum findings start while SFT runs.
3. C-1 → noise floor → C-2/C-3.
4. D-4 to D-8; S-4/S-5; X-1 to X-4.
5. Write `prereg.md` → Phase E → write-up.

## Sources
The per-topic files and their evidence dumps; the main RL recipe evidence is [evidence/06-recipes.md](evidence/06-recipes.md).
- JustRL: arXiv 2512.16649; Tricks or Traps / Lite PPO: arXiv 2508.08221; Qwen2.5-Coder-3B SFT→RL: arXiv 2606.18487
- ScaleRL: arXiv 2510.13786; clip and entropy: arXiv 2509.26114; AceReason-Nemotron 1.1: arXiv 2506.13284
- SFT→RL predictors: arXiv 2510.01624; SFT regimes before RL: arXiv 2512.11470
- LLM4Decompile: arXiv 2403.05286; small-LLM SFT guide: arXiv 2412.13337
