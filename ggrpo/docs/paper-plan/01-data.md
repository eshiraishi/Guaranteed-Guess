# 01. Data: what we can train and evaluate on, and how good each source is

**Status (Oct 24, 2026): plan.** Nothing here has been built. The tags are defined in [00-overview.md](00-overview.md#evidence-tags):
**[V]** verified by the 3-vote check, **[R]** measured in this repo, **[L]** literature not verified in this run, **[P]** our
proposal or inference.

## Summary
- **SFT data needs no tests.** The compiler makes the x86/target pairs, so SFT data is limited only by deduplication,
  decontamination and diversity [P]. GG used 1.31M pairs [R]. Several public corpora with 2–4M functions
  exist [V]. The verified evidence says unique pairs set the ceiling after RL, and that more unique prompts beat
  more responses per prompt [V, math].
- **RL data needs a real oracle, and we can get 40–50x more of it than we have.** Tests belong to a problem, but each accepted C
  solution of a problem compiles to a different x86 prompt. Counting every accepted C solution of problems with official
  tests gives **about 377k real-test RL prompts** before build drops, or **about 43k** at 10 solutions per problem,
  against today's 858 [R].
- **The real-test sources** are Codeforces (open-r1, filtered to complete official tests [V]), AOJ (full judge tests
  downloadable without login [V]) and AtCoder (terms unclear). ExeBench, CodeContests' generated tests and MBPP+ are
  synthetic [V], so they belong to the synthetic arm ([04](04-synthetic-tests.md)).
- **Our evals are too small for the effects RL produces.** A paired test on HumanEval-C (164) detects about 7 pp at
  80% power, on dev (94) about 9 pp, and on clean BringUp (53) about 12 pp [V+R]. The plan therefore builds a
  held-out set of **at least 1,000 problems** with real tests (§5), which brings the detectable effect to about 3 pp
  ([05](05-statistics.md)).

## 1. Tiers
| Tier | Oracle | Use | Examples |
|---|---|---|---|
| **S0 (SFT pairs)** | the compiler output is the label | SFT only | AnghaBench, The Stack C, accepted contest solutions |
| **R (real tests)** | official or human-written tests, with official expected outputs | RL reward, eval | MBPP-C asserts, HumanEval-C asserts, Codeforces/AOJ/AtCoder judge tests |
| **D (reference oracle, real inputs)** | inputs from a real source (judge inputs, fuzz seed corpora); expected outputs from running the reference build | RL arm, eval robustness | the judge inputs of a problem run on another solution; DecompileBench OSS-Fuzz seeds [V] |
| **S (synthetic)** | inputs we generate; expected outputs from the reference build | separate arm ([04](04-synthetic-tests.md)) | AnghaBench/ExeBench functions with generated inputs |

**Rule (Oct 23, kept):** the main RL corpus and every headline eval use tier R. Tier D is close to R, because the
inputs and the oracle are real. It still needs a decision (Decision D3 in §7). Tier S is always reported as its own labeled arm.

## 2. SFT corpora (tier S0)
| Source | Size | Targets available | Notes | Tag |
|---|---|---|---|---|
| GG SFT (`ahmedheakl/gg-armv8-O2-instruct`) | 1,309,052 pairs: 1,037,405 AnghaBench + 271,647 Stack | ARMv8 macOS O2 (other GG releases cover O0, ARMv5, RISC-V) | macOS 13 / SDK 13.3 stamp; C sources of the Stack part in `the_stack_data_c_{0,1,2}` | [R] |
| AnghaBench | ~1M compilable C files | any (recompile) | mined in 2019, before HumanEval and MBPP; GG's main source | [R] |
| SLaDe's expanded AnghaBench | ~4M functions | x86 and ARM, GCC O0/O3 | release and license not verified | [V] |
| Forklift | ~2.7M functions | x86, ARM and RISC-V, Clang and GCC, O0/O3/Oz, plus LLVM IR | the closest existing multi-ISA corpus; release, license and per-ISA split not verified | [V] |
| Decompile-Bench | 2M binary-source pairs from 3,961 permissive Stack v2 repos | Linux x64, Clang-19, O0–O3 (recompile for other targets) | MinHash-LSH dedup; HF card says CC0-1.0 | [V] |
| ExeBench | 4.5M compilable C functions (v1.00 paper) | recompile | its I/O is synthetic (tier S), but the C is usable for SFT | [V] |
| The Stack v2, C | large | recompile | terms of use not verified | [L] |
| Accepted contest solutions (Codeforces, AOJ, AtCoder) | ~377k C programs (§3) | recompile | **keep out of SFT** whenever the problem is in the RL pool or an eval set (§6) | [R] |

**What the SFT redo should use [P]:**
1. **GG's 1.31M programs, rebuilt per target.** Rebuilding reproduces GG's data and makes all targets share the same
   programs: ARMv8 macOS O0/O2, AArch64 Linux O0/O2 and RISC-V64 Linux O0/O2.
2. **New unique programs** from Forklift/ExeBench/Decompile-Bench C, after deduplication against GG's set and
   decontamination against every eval. Unique pairs are what scale the post-RL ceiling (S1K→889K: 77.3→84.0,
   flattening from 220K to 440K) [V, math at 7B]. GG's own data ablation gave 500k→1.3M: 87→98% on HumanEval O0 [R, paper Fig. 5].
3. **Multi-target without multiplying compute.** Each source program appears once per epoch with one target drawn at
   random, and the prompt names the target. Total tokens then stay at GG's level. Per-target models are an ablation ([06](06-schedule.md)).
4. **Dedup:** token-level C dedup plus assembly-level dedup (Forklift), then MinHash-LSH across sources
   (Decompile-Bench) [V]. Reuse our existing 13-gram C and 10-gram asm containment scan (68 s on the Mac) [R].

## 3. RL tasks with real tests (tier R)
### 3.1 Inventory (counted Oct 23–24 from the repo's survey tables)
| Source | Problems | Accepted C solutions | Capped at 10 per problem | Tests | Style | Tag |
|---|---|---|---|---|---|---|
| MBPP-C (decompile-eval) | 952 valid (858 train / 94 dev) | 1 per task | 952 | hand-translated asserts, median 3 | function + driver | [R] |
| Codeforces, complete official tests (stdio, no checker, not interactive) | 890 (862 `executable`) | 32,027 (median 17) | 6,694 | complete official, median 46 | stdin/stdout | [R][V] |
| Codeforces, partial official tests (same filters) | 2,402 | 42,402 (median 5) | 13,614 | untruncated official only; median 40 on the judge, fewer available | stdin/stdout | [R][V] |
| AOJ (Aizu problems in CodeNet) | 1,832 with accepted C | 150,921 | 11,190 | full judge tests from `judgedat.u-aizu.ac.jp` (checked live Oct 24) | stdin/stdout | [R][V] |
| AtCoder (in CodeNet) | 1,391 with accepted C | 152,282 | 11,546 | official tests on Dropbox up to about ABC270, then publication suspended | stdin/stdout | [R] |
| **Total, before build drops** | **~7,500** | **~378k** | **~44k** | | | |

- **Build drops are unknown until we build.** MBPP-C lost 2%. Contest code will lose more: implicit declarations
  are an error in Apple clang 14, and there are GCC extensions, `%I64d`, deep recursion beyond macOS's 8 MB main stack,
  and float printing that differs between platforms [R, data-survey §3.4].
- **The complete-test Codeforces set is biased easy** (median rating 1,200 vs 1,400), because only short tests survive
  untruncated [R].
- **CodeNet's own tests are weak.** Only 51% of problems have test data, and most have a single test [V]. Use the
  judges' official data, not CodeNet's samples.
- **open-r1 'verifiable' is not all real.** It includes problems whose only full oracle is DeepSeek-R1-written
  generators. Filter with `official_tests_complete == True` [V].

### 3.2 Why several solutions per problem is valid, and what it costs [P]
- The task's input is the x86 program. Two accepted solutions of one problem are two different x86 programs with the
  same specification, so the official tests are a correct oracle for both.
- **Correlation:** prompts of one problem share their tests and often their algorithm. Train/dev/test splits must be
  **by problem**, and the statistics must cluster on the problem ([05](05-statistics.md)).
- **Diversity:** near-identical solutions add little. Dedup within each problem on normalized x86 (the 10-gram
  containment from the contamination scan), then cap at k solutions per problem. The choice k ∈ {1, 3, 10, all} is a data-scale
  experiment in its own right ([06](06-schedule.md) D-2).
- **Length:** AOJ median solution length is 926 characters and AtCoder 696 [R]. Whole-program x86 at O2 is longer than
  MBPP functions, so filter by reference ARM tokens against the completion cap ([02](02-harness.md)).

### 3.3 Not real (tier S), kept out of the main corpus
ExeBench I/O (type-guided random inputs, reference outputs) [V]; MBPP+ (LLM seeds plus type-aware mutation) [R];
CodeContests generated tests and CodeContests+ [R]; open-r1 generated tests and `generated_checker` [V][R]; TransCoder
GfG (auto inputs, harness bugs) [R]; rStar-Coder, KodCode and other LLM-written sets [R]; YARPGen/Csmith (random
programs) [R]. Several of these are inputs to the synthetic arm.

### 3.4 Tier D candidates
- **The judge inputs of a problem run on a new solution** are already tier R, because the expected outputs are official.
- **DecompileBench (arXiv 2505.11340):** 23,400 functions from 130 OSS-Fuzz projects, each with the project's seed corpus,
  and correctness judged by equal whole-program branch profiles [V]. This is real-world, non-contest code with real
  inputs. The cost is that each project's build has to be reproduced. It is a good **robustness eval** for "does RL on contest code
  transfer to library code" [P].

## 4. Function-level vs whole-program
- MBPP-C and HumanEval-C are function-level: the prediction links against a C driver that calls `func0`.
- Contest data and BringUp are whole programs with `main`, stdin and stdout.
- Run 1 improved HumanEval-C but not BringUp [R]. The contest data sits between the two, with whole programs of
  moderate length. That makes it the natural bridge, and it is a curriculum axis ([03](03-curriculum.md)).

## 5. Evaluation sets
| Set | Size | Oracle | Role | MDE, paired, 80% power | Tag |
|---|---|---|---|---|---|
| HumanEval-C O2 (GG bench + authors' tests) | 164 (152 without MBPP twins) | authors' asserts | **comparability with GG**; reported | ~7 pp | [R][V] |
| BringUp O2 | 65; **53 clean** | upstream program tests | whole-program generalization; supporting | ~11–12 pp | [R][V] |
| MBPP-C dev | 94 | asserts | selection only | ~9 pp | [R][V] |
| **GG-XL held-out (new)** | **≥1,000 problems**, 1–3 solutions each | official judge tests | **primary endpoint** | ~3 pp (needs measured ω²) | [P] |
| DecompileBench subset | a few hundred functions | seed-corpus profiles | robustness to real code | — | [P] |
| Time-split set | TBD | real tests | contamination control | — | open (§7) |

**GG-XL construction [P]:**
- Draw problems from the AOJ and Codeforces pools **before** any RL split. Stratify by judge, rating or volume, and by
  reference length.
- Hold out roughly 15–20% of problems, about 1,000–1,300. Never put their solutions in SFT or RL.
- Use one to three solutions per problem, and analyze clustered by problem.
- The MDE comes from Miller's formula ([05](05-statistics.md)), MDE = 2.80·sqrt(ω²/n). With our measured HumanEval
  ω² ≈ 0.095, n = 1,000 gives about 2.7 pp.

**Time split.** Codeforces accepted C stops in 2024 (one problem from 2024) [R]. The open-r1 test split (468 problems,
contests 1993–2067, late 2024 to early 2025) has tests but needs reference programs. The options are C++ submissions,
which give long x86 because of STL, or C written by us. That is open question Q4 in §7.

## 6. Contamination and leakage control
- **Done:**
  - BringUp: 12/65 programs are copies of SFT files, so the clean set is 53.
  - HumanEval-C: clean.
  - 13 MBPP tasks are twins of 12 HumanEval problems.
  - Qwen2.5-Coder's decontamination matched only Python 10-grams. [R]
- **For every new source [P]:**
  1. Run the C 13-gram and asm 10-gram containment scans against GG SFT, all evals and the RL pool.
  2. Run a statement-level same-problem check (TF-IDF plus embeddings plus manual review of the top pairs) between
     contest problems and HumanEval/MBPP/BringUp.
  3. Keep RL and eval problems out of SFT by default. "RL problems also in SFT" is a separate arm (it is the SFT
     data-control question from Oct 23).
  4. Write an exclusion file, with a reason per ID, that `data.py` reads. This is the toggle proposed in data-survey §2.5.
- **Base-model contamination can't be fixed.** It affects base and trained models alike, so the gains stay meaningful and
  the absolute scores may run high [R]. A new base model ([06](06-schedule.md) B0) needs its own probe, for example a
  perplexity comparison of eval references against unseen code.

## 7. Decisions and open questions
**Licenses and terms: deferred (Enzo, Oct 24).** They are decided per source when we decide what to run. Nothing here
was confirmed by the 3-vote check except ExeBench (its own contributions are MIT; functions keep their repositories'
licenses) and Decompile-Bench (HF card CC0-1.0) [V].

| Source | What we know | Status |
|---|---|---|
| open-r1/codeforces | CC-BY-4.0 in metadata vs ODC-By in card text [R] | decide; also check Codeforces' own terms for training on tests |
| AOJ judge data | public, undocumented endpoint; terms not checked [V] | **ask AOJ** or read terms before the crawl |
| AtCoder tests | publication suspended because of misuse [R] | **ask AtCoder** |
| Project CodeNet | Apache-2.0 [R] | covers the submissions, not the judges' test data |
| Forklift, SLaDe corpus | release and license not verified [V] | check |
| The Stack v2 | terms of use not verified | check |
| BringUp, HumanEval-C, MBPP-C | already in use | — |

**Decisions:**
- D1. **Decided (Oct 24): several solutions per problem, with a dedup cap.** The value of k is experiment D-2 in
  [06](06-schedule.md); k = 10 until then.
- D2. Use the partial-test Codeforces problems, with a minimum of at least 5 official tests?
- D3. Do tier-D tasks count as real for the main corpus?
- D4. Include C++ solutions with a length filter? This is the only route to post-2024 contest problems.
- D5. Keep RL and eval problems out of SFT (proposed: yes)?

**Open questions:**
- Q1. How many contest programs survive the macOS 13 / clang 14 build, and how many survive the Linux and RISC-V builds?
- Q2. What are base GG's pass rates on contest programs? The census in [06](06-schedule.md) A-4 measures them.
- Q3. Are the Forklift or SLaDe corpora released? Either would save rebuilding for RISC-V.
- Q4. What is the time-split eval source?

## Sources
Verified claims with quotes and votes: [evidence/01-data.md](evidence/01-data.md). Repo measurements:
`docs/data-survey.md`, `results/data-survey/realio/`.
- open-r1 Codeforces: https://huggingface.co/datasets/open-r1/codeforces
- online-judge-api-client (AOJ system cases): https://github.com/online-judge-tools/api-client
- Project CodeNet: https://github.com/IBM/Project_CodeNet, arXiv 2105.12655
- ExeBench: https://huggingface.co/datasets/jordiae/exebench, DOI 10.1145/3520312.3534867
- Decompile-Bench: arXiv 2505.12668; DecompileBench (OSS-Fuzz): arXiv 2505.11340
- SLaDe: arXiv 2305.12520; Forklift: arXiv 2404.16041; D-LiFT: arXiv 2506.10125; Echo: arXiv 2609.18706
- LLM4Decompile: https://github.com/albertan017/LLM4Decompile
- SFT data scale vs RL ceiling: arXiv 2512.11470; prompts vs responses scaling: arXiv 2506.13284
- GG: arXiv 2506.14606 (repo PDF)
