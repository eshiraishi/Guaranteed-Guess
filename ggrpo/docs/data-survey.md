# Data survey: contamination audit and training data with real tests

**Status (Oct 23, 2026): research only.** Nothing here is implemented. No task, split or run has been changed.
This doc collects what we found so the data plan can be written from it. Open decisions are listed in §5, the
evidence files in §6 and the sources in §7.

## Summary
- **Nothing we built or trained is lost.**
  - Every finding below leads to an exclusion list or a reporting split; no task needs rebuilding.
  - Run 1's HumanEval gain holds without the MBPP twins: 65 → 79 on the 152 non-twin problems, against 72 → 87 on
    all 164.
- **HumanEval-C is clean** against GG's SFT data. Only 7 one-line functions compile to the same assembly as some
  SFT function.
- **BringUp is not clean: 12 of its 65 programs are copies of files in GG's SFT data.**
  - The paper counts 3 of the 12 as passes.
  - Report BringUp on all 65 and on the clean 53. Base GG scores 6/65 and 6/53 at beam 8.
- **13 MBPP-C tasks are the same problems as 12 HumanEval problems.** 9 are in train and 4 in dev.
  - Either drop them from train/dev, which keeps HumanEval at 164,
  - or keep them and report HumanEval on 152.
- **Contamination of Qwen2.5-Coder's pre-training data can't be checked or fixed.** Its decontamination matched
  Python 10-grams, which misses C translations.
- **New training data must have real tests** (decision, Oct 23). Three sources qualify. All three are whole
  stdin/stdout programs with accepted human C solutions:

  | Source | Problems | Tests | Blocker |
  |---|---|---|---|
  | Codeforces | 890 | complete official test set | none |
  | Codeforces, partial | 2,402 more | only the short official tests | needs a decision |
  | AOJ (Aizu problems in CodeNet) | up to 1,832 | full official tests from the judge's API | needs a crawl |
  | AtCoder (in CodeNet) | 1,391 | official tests, on AtCoder's Dropbox only | terms unclear |

- **Every other source we looked at has synthetic tests or no tests.** That covers ExeBench, MBPP+, the generated
  tests of CodeContests and open-r1, CodeContests+, SuperCoder, TransCoder GfG, YARPGen/Csmith and LLM-written sets.
  CodeContests has no real tests that open-r1 lacks.
- **The new data needs a stdio reward mode** and the same build standard as MBPP-C. A pilot on the 890 Codeforces
  problems is the cheapest next step.

## 1. Current data and the task standard

| Set | Role | Source | Tasks | Tests |
|---|---|---|---|---|
| MBPP-C | train 858 / dev 94 | `LLM4Binary/decompile-eval` @ b9271fae, `mbpp`, C [1] | 952 valid (974 upstream, 973 built) | MBPP's asserts, translated to C by hand by the LLM4Decompile authors [2]; median 3 per task |
| HumanEval-C | eval | `ahmedheakl/gg-bench-armv8-O2` + `ahmedheakl/asmbench_source` [3] | 164 | the GG authors' asserts; median 6 per task |
| BringUp | eval | `ahmedheakl/gg-bench-bringup-O2`, upstream bringup-bench [4] | 66 rows; 65 have upstream source (`satomi` does not) | whole programs |

MBPP-C task IDs are `mbpp/{rank}`, where rank is the row order after sorting by `index`. Rank i is MBPP task_id
i+1 (verified).

**Task standard.** Every current train/dev task meets all of these:
1. The C compiles standalone with system headers only.
2. It is built with GG's commands at -O2 in the gg-ventura VM (macOS 13.6, Apple clang 14.0.3, SDK 13.3).
3. The test driver has real oracles: the dataset's asserts, not generated ones.
4. The reference ARM passes.
5. An empty `.s` fails to link (`neg_verdict = "link_err"`).
6. No driver defines `func0`.
7. Each prompt is one x86 compilation unit.

## 2. Contamination audit (Oct 22–23)

### 2.1 What was checked against what

| Probe set | Compared with | Level | Method |
|---|---|---|---|
| HumanEval-C (164), BringUp (65), MBPP-C (952) | C source of all 271,647 Stack files in GG's SFT data | C | canonical-token 13-gram containment |
| the same three sets | x86 of all 1,309,052 SFT rows (AnghaBench + Stack) | assembly | normalized-instruction 10-gram containment |
| MBPP-C train/dev | HumanEval-C | problem | statement TF-IDF, C 5-gram Jaccard, embeddings, then manual review |

**GG's SFT data** (`ahmedheakl/gg-armv8-O2-instruct` [5]):
- 1,309,052 rows with columns `x86`, `arm`, `file` and `source`: 1,037,405 AnghaBench rows and 271,647 Stack rows.
- The paper reports 1.01M AnghaBench programs and 306k Stack programs, and mentions deduplication within the training
  data only, with no decontamination against HumanEval or BringUp [6].
- All rows carry the macOS 13 / SDK 13.3 stamp, so assembly matches against our builds are meaningful.

**C sources of the Stack part:** `ahmedheakl/the_stack_data_c_{0,1,2}` [5] hold 327,904 files. Matching by file name,
they cover all 271,647 SFT Stack files.

**The AnghaBench part was checked at the assembly level only.** AnghaBench was mined in 2019 [7], before HumanEval
(2021) [8] and MBPP (2021) [9] existed, so it can share only generic functions such as `gcd`.

### 2.2 Method
- **C level:**
  - Comments and preprocessor lines are stripped.
  - String literals become `S`, char literals `C`, and identifiers outside a keyword/libc list `I`.
  - Each 13-token window is hashed (u64 polynomial).
- **Assembly level:**
  - `##` comments and directives are dropped; labels become `L` and symbols `_S`. Each instruction is one token.
  - Functions are split at `-- Begin function`, and each 10-instruction window is hashed.
- **Boilerplate removal:** shingles that occur in ≥20 probes or in >1000 corpus documents are dropped.
- **Score:**
  - Containment = shared shingles / probe shingles, keeping the top 3 corpus documents per probe.
  - Every BringUp or HumanEval hit at ≥0.33 was read by hand. A BringUp program counts as copied when the reading
    confirms it, or when it has an exact x86 function match of ≥30 instructions.
- **Same problem (MBPP vs HumanEval):**
  - Original MBPP statements were compared with HumanEval docstrings by TF-IDF, and the C by canonical 5-gram
    Jaccard.
  - Then `BAAI/bge-small-en-v1.5` embeddings [10] gave the top 3 MBPP tasks per HumanEval problem. They caught
    synonyms TF-IDF missed, such as "toggle case" vs `flip_case`.
  - The top pairs were reviewed by hand.
- **Cost:** the full scan takes 68 s on the Mac (polars). Similar methods in the literature are 13-gram plus
  embedding plus AST checks (phi-1 [11]) and BigCode's substring and MinHash tools [12].

### 2.3 Results

**HumanEval-C: clean.**
- No C copies in the SFT Stack files.
- At the assembly level, only one-line idioms match some SFT function exactly: problems 13, 23, 35, 41, 42, 48 and
  53. Examples are `strlen`, `gcd` and "add 1 to each element". There is one natural way to write each of them. No
  action needed.

**BringUp: 12 of 65 programs are copies of SFT Stack files.** Most are classic programs that are copied all over
GitHub: the a1k0n donut, MT19937, fix_fft, TheAlgorithms/C and GfG.

| Copied program | Counted as a pass in the paper |
|---|---|
| donut, spirograph, fy-shuffle | yes |
| qsort-demo, fft-int, knights-tour, mersenne, c-interp, max-subseq, cipher, graph-tests, boyer-moore-search | no; these could flip after RL for the wrong reason |

- **Seven more programs share one textbook routine with SFT files.** These are shortest-path, blake2b, dhrystone,
  murmur-hash, topo-sort, priority-queue and knapsack, sharing for example Floyd–Warshall or the BLAKE2 constant
  table. They stay in the clean set.
- **Base GG:**

  | Decoding | All 65 | Clean 53 | Copied programs among the passes |
  |---|---|---|---|
  | beam 8 | 6 | 6 | 0 |
  | greedy | 5 | 5 | 0 |

**MBPP-C vs SFT.**
- At the C level, 25 of the 865 tasks large enough to score have containment ≥0.8. At the assembly level, 76 tasks
  do.
- GG has seen these prompts. That does not make RL on them invalid; they just teach less. Keep them, with an
  `in_sft` flag.

**MBPP-C vs HumanEval-C: 13 same-problem twins.**

| MBPP task | Split | HumanEval | HumanEval function |
|---|---|---|---|
| mbpp/686 | train | 13 | greatest_common_divisor |
| mbpp/812 | train | 23 | strlen |
| mbpp/556 | dev | 27 | flip_case |
| mbpp/499 | dev | 28 | concatenate |
| mbpp/604 | dev | 31 | is_prime |
| mbpp/872 | train | 55 | fib |
| mbpp/67 | train | 57 | monotonic |
| mbpp/1 | train | 58 | common |
| mbpp/31 | train | 59 | largest_prime_factor |
| mbpp/32 | train | 79 | decimal_to_binary |
| mbpp/98 | dev | 79 | decimal_to_binary |
| mbpp/754 | train | 90 | next_smallest |
| mbpp/957 | train | 156 | int_to_mini_roman |

- **Close variants, kept:** eight MBPP tasks share the loop shape of a HumanEval problem but have a different
  condition: mbpp/255→96, 269→121, 566→126, 630→140, 65/312/851→30 and 460→98. Keeping them is ordinary
  generalization.
- **No published measurement of MBPP–HumanEval overlap exists** that we could find.

**Base model (cannot be fixed).**
- Qwen2.5-Coder removed HumanEval, MBPP, GSM8K and MATH with a 10-gram word-overlap filter [13]. That filter matches
  Python text, so C translations pass through it.
- Translated and rephrased benchmarks evade n-gram filters and still inflate scores [14].
- Riddell et al. found 18.9% of HumanEval and 20.8% of MBPP solutions in The Stack's Python split [15].
- This affects the base model and the RL model equally, so gains stay meaningful, but absolute scores may run high.

### 2.4 Effect on existing data and on run 1
- **Data:** no task needs rebuilding.
- **Run 1** trained on all 858 train tasks, including the 9 train twins. HumanEval-C at beam 8, base vs
  checkpoint-106:

  | HumanEval subset | Problems | Base | Checkpoint 106 | Up / down |
  |---|---|---|---|---|
  | all | 164 | 72 | 87 | 16 / 1 |
  | without the 9 train twins | 155 | 66 | 81 | 16 / 1 |
  | without all 12 twin problems | 152 | 65 | 79 | 15 / 1 |

  The only twin problem whose result changed is humaneval/28. Its twin, mbpp/499, is in dev, which run 1 never
  trained on. So the twins did not inflate run 1.
- **Dev scores are unaffected,** since dev is MBPP-C.

### 2.5 Measures
1. **Exclusion file with an on/off switch** (proposed, not implemented):
   - A one-off deterministic script writes the excluded task IDs, each with its reason, to a file in `data/`.
   - `data.py` reads it the same way it reads `settings.data.train_ids`. Leaving the path unset turns exclusion off.
   - The first entries are the 13 twins. The script only needs rerunning when a data source is added.
2. **Report BringUp on 65 and on 53.**
3. **Report HumanEval on 164.** For any checkpoint trained with the twins (run 1), also report it on 152.
4. **Flag MBPP tasks with SFT containment ≥0.8 as `in_sft`,** and keep training on them.
5. **Check every new source the same way:**
   - Build it in the VM.
   - Scan it against the SFT data, and use the result to pick a solution that isn't flagged.
   - Scan it against HumanEval-C and BringUp, and exclude matches.
   - Run the same-problem check on its statements.
6. **Optional, not run:**
   - HumanEval-C with renamed identifiers and changed constants: see whether pass@1 drops (about 1 h).
   - A loss probe of GG on the HumanEval reference ARM, compared with training rows and unseen code (about 15 min on
     one GPU).

### 2.6 Limits
- **Heavily rewritten copies can be missed.** Containment catches copies and light edits, but a heavy rewrite
  changes both the tokens and the assembly.
- **Twins worded very differently can be missed,** since only the top candidate pairs were reviewed.
- **AnghaBench C was not scanned,** only its assembly.
- **The Stack version GG used is unverified.** The paper says "Stackv2" but cites and describes v1 [6, 16].

## 3. Training data with real tests

### 3.1 Rule (Oct 23)
"We need data that has real I/O cases in the same way we used so far. Doing synthetic cases is something for another
paper later on." Changes to the harness are allowed.

**Working definition:**
- **Real:** the inputs were chosen by the benchmark or judge authors, and the expected outputs are the official ones.
- **Official judge tests count as real.** Problem setters often make the large tests with their own generator
  scripts, but the result is the judge's official, validated test set.
- **Synthetic:** any test whose inputs we, a third party or a model generate (by mutation, at random or with an LLM),
  even when a reference solution produces the outputs.

### 3.2 Sources that qualify

**Codeforces, from open-r1**
- **Data:**
  - `open-r1/codeforces` [17]: 10,024 problems. The license is CC-BY-4.0 in the metadata but ODC-By in the card text.
  - `open-r1/codeforces-submissions` [17]: 12.59M submissions with source code.
- **Tests:**
  - Codeforces truncates long tests to about 400 characters, and `official_tests` keeps only the untruncated ones.
  - `official_tests_complete` marks the 1,892 problems where nothing was truncated, so the full official set is
    present.
  - `executable` means at least 3 human solutions pass the official tests.
- **C submissions (counted Oct 23):**
  - 110,657 C rows; 85,535 accepted (GNU C 49,952 and GNU C11 35,583) across 4,083 problems.
  - The `selected_accepted` config has no C.
- **Funnel:**

  | Filter | Problems | Official tests (median) | Accepted C (median) | Rating (median) |
  |---|---|---|---|---|
  | has accepted C | 4,078 | 41 | 6 | 1,400 |
  | + stdio, no `generated_checker`, not interactive | 3,292 | 41 | 7 | 1,400 |
  | **+ complete official tests** | **890** (862 `executable`) | 46 | 17 | 1,200 |
  | + only partial official tests | 2,402 | 40 on the judge; fewer available | 5 | 1,400 |

- **Excluded within the source:**
  - `generated_checker`: an LLM-written checker, so synthetic.
  - Interactive problems, and the 40 with file I/O.
  - The ~110 GB of generated tests, which are synthetic.
- **The complete set is biased toward small inputs.** All of its tests are short, so these problems tend to be
  easier (median rating 1,200 vs 1,400).
- **Partial sets:** in one CodeContests shard, 207 Codeforces problems have a median of 14 untruncated official tests,
  and 71% have at least 5 (§3.3).
- **No post-cutoff set is possible.** Problem years with accepted C run from 2010 to 2024, and only one problem is
  from 2024, so a post-cutoff eval set would need another source.

**AOJ (Aizu problems in CodeNet)**
- **Data:**
  - CodeNet [18] is Apache-2.0. Its Aizu problems are IDs p00000–p02477.
  - 1,832 problems have accepted C: 150,921 accepted C submissions, a median of 6 per problem, and 1,062 problems
    with at least 5.
  - These counts come from the HF mirror `iNeil77/CodeNet` (C subset: 729,296 submissions, 303,203 accepted), because
    IBM's CDN did not respond on Oct 23.
  - Aizu submissions are labelled `C`; AtCoder ones are `C (GCC …)` or `C (Clang …)`.
  - An earlier count from the official metadata gave 313,360 accepted C over 3,265 problems, consistent with the
    1,832 + 1,391 = 3,223 problems here.
- **Tests:** AOJ's judge-data API [19] serves the official cases.
  - `GET https://judgedat.u-aizu.ac.jp/testcases/{id}/header` lists the cases with their sizes.
  - `.../testcases/{id}/{serial}/in` and `/out` return the full files. Checked on ALDS1_4_B case 10: all 866,743
    input bytes arrived.
  - `.../testcases/{id}/{serial}` (JSON) cuts large cases off ("terminated because of the limitation"). Don't use it.
  - `.../testcases/samples/{id}` returns the statement samples.
  - The old problem-list API (`judgeapi.u-aizu.ac.jp/problems`) returns 410 Gone.
- **ID mapping needs a crawl.**
  - CodeNet renumbers problems: p00481 is AOJ 0558 and p02343 is DSL_1_A. Mapping CodeNet number n to AOJ n matched
    0 of 30 sampled problems.
  - Matching CodeNet's sample I/O (from `Elfsong/codenet_metadata` or the CodeNet problem HTML) against
    `/testcases/samples/{id}` works.
  - That means crawling the AOJ ID space: the numeric volume IDs plus course IDs such as `ITP1_*`, `ALDS1_*`, `DSL_*`,
    `GRL_*`, `CGL_*`, `NTL_*` and `DPL_*`.
- **Test coverage:**
  - In 30 AOJ volume problems, effectively random because they were picked by the failed numeric mapping, 4 had no
    judge data and 10 had a single test file; older volumes put many cases in one file. The rest had 2–137 files.
  - The API does not flag special judges. The exact-compare reference check (§3.4) drops them.
- **Effort:** about 3–4k requests for the mapping, plus roughly 1,800 problems × 10–30 files × 2 requests for the
  tests. That is tens of thousands of requests, 2–5 h at a polite rate.
- **AOJ's terms for bulk use of judge data:** not checked.

**AtCoder (in CodeNet)**
- **Data:** 1,391 problems (p02534–p04051) have accepted C: 152,282 accepted submissions, a median of 32 per problem,
  and 1,146 problems with at least 5.
- **Tests:** AtCoder published official test cases on Dropbox up to about ABC270, then suspended publication because
  of misuse [20]. Whether they may be used for training is unclear, so ask AtCoder first.
- **Mapping:** CodeContests names these problems with the CodeNet ID and the contest name (for example
  `p02613 AtCoder Beginner Contest …`), which gives the contest/task mapping.

**CodeNet sample I/O (fallback only)**
- These are real but few: in the CodeContests shard, a median of 3 per problem for AtCoder and 1 for Aizu.
- Too weak on their own; usable as extra tests.

### 3.3 Sources that do not qualify

**CodeContests [21] (measured on shard `train-00000`, 342 problems):**

| Source | Problems | Public (median) | Private (median) | Problems with private tests | With ≥5 private | Generated (median) |
|---|---|---|---|---|---|---|
| Codeforces | 207 | 2 | 14 | 86% | 71% | 63 |
| Aizu | 51 | 1 | 0 | 0% | 0% | 100 |
| AtCoder | 30 | 3 | 0 | 0% | 0% | 100 |
| HackerEarth | 30 | 1 | 1 | 73% | 7% | 100 |
| CodeChef | 24 | 1 | 0 | 0% | 0% | 100 |

- Solution languages are only Python 2, C++, Python 3 and Java. There is no C.
- Its Codeforces private tests are the untruncated official tests, which open-r1 already has.
- Its AtCoder and Aizu rows come from CodeNet and have only statement samples plus generated tests.

**Other sources:**

| Source | Size | Why it doesn't qualify | Ref |
|---|---|---|---|
| ExeBench `train_synth_rich_io` | 107,250 functions × 30 IO pairs | random seeded inputs; outputs recorded on Linux by running the function itself; GCC/Linux assembly. `train_real_simple_io` is trivial (53.8% give the same output for every input). Scraped from GitHub, so high SFT overlap | [22] |
| MBPP+ | 378 tasks (365 overlap our train/dev without the twins); median 105 extra inputs | inputs generated by LLM seeding plus type-aware mutation | [23] |
| open-r1 generated tests / `generated_checker` | ~110 GB / ~30% of problems | LLM-written generators and checkers | [17] |
| CodeContests+ | 11,690 problems | inputs come from generators | [24] |
| SuperCoder | 7,872 + 200 CodeNet C programs | CodeContests generated inputs, with outputs from rerunning the submission | [25] |
| xCodeEval | 7,514 Codeforces problems | CC BY-NC 4.0 on GitHub vs CC-BY-4.0 on HF; only ~1.3k held-out problems have complete tests; overlaps open-r1 | [26] |
| TransCoder GfG C | 698 scripts | 10 auto-generated inputs; harness bugs (only 2 inputs actually tested, results never printed) | [27] |
| YARPGen / Csmith / ldrgen | unbounded | random programs; unnatural code; Csmith compiles to trivial assembly | [28] |
| LLM-written sets (rStar-Coder, KodCode, OSS-Instruct) | — | synthetic tests, and no C | [29] |
| Jotai | ~30k functions | AnghaBench functions (GG's own SFT source); inputs from constraints; no stored outputs | [30] |
| DecompileBench, Decompile-Bench GitHub2025, Idioms, Guess & Sketch | — | needs whole Linux project builds, has no tests, or is too small (72 programs) | [2, 30] |

This replaces two Oct 22 suggestions: MBPP+ inputs and CodeContests generated tests are out under the new rule.

### 3.4 What the new tasks need
- **Harness: a stdio reward mode.**
  - Link the predicted `.s` alone (no driver) and run each official test with its input on stdin.
  - Compare stdout exactly, ignoring only trailing whitespace on each line and at the end of the file.
  - Use a per-test timeout. Today `run_timeout` is 10 s per run.
- **The task standard for whole programs:**

  | Current criterion | Whole-program version |
  |---|---|
  | compiles standalone, system headers only | the same |
  | built at -O2 in the VM with GG's commands | the same |
  | the driver has real oracles | official expected outputs |
  | the reference ARM passes | the native ARM build passes every official test, twice (determinism) |
  | an empty `.s` fails to link | holds, because there is no `_main` |
  | no driver defines `func0` | not applicable (no driver) |
  | one x86 compilation unit | the same |
  | — | **new:** at least 2 distinct expected outputs, so a program that prints a constant cannot pass |
  | — | **new:** the reference ARM fits `max_completion_length` (4096 tokens), and the prompt fits the SFT length distribution |

- **The build will drop programs, without manual work, that:**
  - use implicit declarations (an error in Apple clang 14) or GCC-only extensions;
  - use `%I64d`;
  - need more than the 8 MB main-thread stack macOS gives by default (Codeforces allows 256 MB);
  - print floats that differ on macOS;
  - need special judges, or time out.
- **How many survive is unknown until we build them.** MBPP-C lost 2%; contest code will lose more.
- **Solution choice:** most problems have several accepted C solutions. Take one per problem by default, preferring
  one the SFT scan doesn't flag. More than one per problem is a possible data-scale ablation.
- **Split:** train/dev by problem, never by solution. Keep the current MBPP dev for comparability, and consider a
  separate dev set from the new source.
- **Reward cost:**
  - A median of 41–46 test runs per rollout, against one driver run today.
  - Option: cap the tests used in training (for example 20, fixed once per problem) and run the full set on dev.
- **Domain shift:** these are whole programs with `main` and scanf/printf, closer to BringUp than to the
  function-level HumanEval.

### 3.5 Yield and effort

| Step | Problems | Effort | Blocker |
|---|---|---|---|
| stdio reward mode | — | about 0.5–1 day of code | none |
| Codeforces, complete tests | 890, minus build drops | download the problems (~2.7 GB parquet, tests included) and the C rows of the submissions; VM build about 1 h | none |
| Codeforces, partial tests | 2,402 | same as above | minimum number of tests |
| AOJ | ≤1,832, minus problems without judge data (4 of 30 in a sample), minus build drops | crawl 2–5 h, then the VM build | AOJ terms not checked |
| AtCoder | ≤1,391 | Dropbox download | AtCoder terms |

**Totals:**
- With complete official tests only: 890 Codeforces problems plus roughly 1,500 AOJ problems is about 2.4k problems,
  about 2.5× the 939 MBPP-C tasks left after the twins.
- With the partial Codeforces problems and AtCoder, up to about 6k.
- Both totals come before build drops.

### 3.6 Suggested order
1. Build the stdio reward mode, then pilot it on the 890 complete Codeforces problems. Measure:
   - how many survive the build;
   - prompt and reference lengths against the SFT distribution;
   - SFT overlap;
   - GG's pass rate. GRPO gets no signal from problems GG always passes or always fails.
2. If the pilot looks good, run the AOJ mapping and test crawl.
3. Use AtCoder only after its terms are settled.
4. Add the partial-test Codeforces problems if we accept a minimum test count. At least 5 official tests would still
   beat MBPP's median of 3 asserts.

## 4. Things that were checked and ruled out
- **Mapping CodeNet IDs to AOJ IDs by number** (0 of 30 matched; see §3.2).
- **AOJ's JSON test endpoint,** which cuts off large cases.
- **CodeContests as a source of real AtCoder/Aizu tests:** there are 0 private tests.
- **open-r1 `selected_accepted` as a source of C:** it has none.
- **A post-2024 Codeforces C eval set:** the accepted C stops in 2024.

## 5. Open decisions
1. Drop the 13 twins, keeping HumanEval at 164? Or keep them and report on 152?
2. Implement the exclusion file with its on/off switch now (§2.5)?
3. Accept official judge tests as real, including large tests that setters generate (§3.1)?
4. Use the partial-test Codeforces problems, and with what minimum number of tests?
5. Run the AOJ crawl (2–5 h)?
6. Ask AtCoder about using its test cases?
7. Cap the tests per rollout in training, and at how many?
8. One solution per problem, or several?

## 6. Evidence files
- `results/data-survey/`: outputs of the contamination audit.
  - `decontam.json`: final lists (copied BringUp programs, partial programs, twins, variants, trivial HumanEval).
  - `neardup_{c,asm}{,_size}.parquet` and `neardup.log`: the near-duplicate scan.
  - `sameprob_{pairs,candidates,embed}.parquet` and `sameprob.txt`: the same-problem pairs.
  - `neardup_runs.parquet`: the longest contiguous matched region for each hit.
- `ops/mac-scratch/9da06990/scripts/`: the audit scripts (`neardup.py`, `neardup_lib.py`, `sameprob.py`, `embed.py`,
  `runs.py`, `show.py`).
- `results/data-survey/realio/`: the real-test survey.
  - `cf_problems.parquet`: 10,024 Codeforces problems with filter columns and accepted-C counts.
  - `codenet_c_ac.parquet`: CodeNet problems with accepted C, per judge.
  - `scripts/`: `cf_count2.py` (problem metadata), `cf_rest.sh` + `cf_project.py` (submission shards), `cf_join.py`
    (funnel), `codenet_count.py`, `cc_sources.py` (CodeContests per source), `aoj_map.py` (the failed numeric
    mapping), `schema.py`, `realio_fetch.sh`.
  - Anonymous HF downloads hit rate limits (HTTP 429), so `cf_rest.sh` fetches whole shards one request at a time,
    with retries.

## 7. References
1. LLM4Decompile, decompile-eval: https://huggingface.co/datasets/LLM4Binary/decompile-eval (CC0-1.0);
   https://github.com/albertan017/LLM4Decompile; Tan et al., "LLM4Decompile", arXiv 2403.05286.
2. Decompile-Bench (MBPP/HumanEval C made by hand; GitHub2025 set): arXiv 2505.12668.
3. GG eval data: https://huggingface.co/datasets/ahmedheakl/gg-bench-armv8-O2,
   https://huggingface.co/datasets/ahmedheakl/asmbench_source.
4. BringUp-Bench: https://github.com/toddmaustin/bringup-bench.
5. GG SFT data: https://huggingface.co/datasets/ahmedheakl/gg-armv8-O2-instruct; Stack C sources:
   https://huggingface.co/datasets/ahmedheakl/the_stack_data_c_0 (also `_1`, `_2`).
6. Heakl et al., "Guaranteed Guess", EMNLP Findings 2025 (`2025.findings-emnlp.1330.pdf` in the repo root);
   arXiv 2506.14606, §3.1.
7. AnghaBench: https://github.com/brenocfg/AnghaBench.
8. Chen et al., HumanEval, arXiv 2107.03374.
9. Austin et al., MBPP, arXiv 2108.07732.
10. BAAI/bge-small-en-v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5; fastembed: https://github.com/qdrant/fastembed.
11. Gunasekar et al., phi-1, arXiv 2306.11644.
12. BigCode decontamination and MinHash tools: https://github.com/bigcode-project/bigcode-dataset; Dolos:
    https://github.com/dodona-edu/dolos.
13. Hui et al., Qwen2.5-Coder technical report, arXiv 2409.12186.
14. Yang et al., "Rethinking benchmark and contamination for language models with rephrased samples", arXiv 2311.04850.
15. Riddell et al., "Quantifying contamination in evaluating code generation capabilities of language models",
    ACL 2024: https://aclanthology.org/2024.acl-long.761. See also Matton et al., EMNLP Findings 2024, arXiv 2407.07565.
16. The Stack: Kocetkov et al., arXiv 2211.15533; https://huggingface.co/datasets/bigcode/the-stack;
    https://huggingface.co/datasets/bigcode/the-stack-v2.
17. open-r1 Codeforces: https://huggingface.co/datasets/open-r1/codeforces,
    https://huggingface.co/datasets/open-r1/codeforces-submissions.
18. Puri et al., Project CodeNet, arXiv 2105.12655; https://github.com/IBM/Project_CodeNet (Apache-2.0). HF mirrors:
    https://huggingface.co/datasets/iNeil77/CodeNet, https://huggingface.co/datasets/Elfsong/codenet_metadata.
19. AOJ judge data API: https://judgedat.u-aizu.ac.jp/testcases/{problemId}/header.
20. AtCoder notice on test-case publication: https://atcoder.jp/posts/1377.
21. Li et al., AlphaCode / CodeContests, Science 2022, arXiv 2203.07814; https://huggingface.co/datasets/deepmind/code_contests
    (CC-BY-4.0).
22. Armengol-Estapé et al., ExeBench, MAPS 2022, DOI 10.1145/3520312.3534867; https://huggingface.co/datasets/jordiae/exebench;
    ExeBench-FHD: arXiv 2610.28212, https://huggingface.co/datasets/davish007/ExeBench-FHD; Decaf: arXiv 2605.11501.
23. Liu et al., EvalPlus, arXiv 2305.01210; https://github.com/evalplus/mbppplus_release.
24. CodeContests+: https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus.
25. SuperCoder: arXiv 2505.11480.
26. Khan et al., xCodeEval, arXiv 2303.03004; https://huggingface.co/datasets/NTU-NLP-SG/xCodeEval;
    https://github.com/ntunlp/xCodeEval.
27. TransCoder GfG C tests: https://github.com/facebookresearch/CodeGen (`data/transcoder_evaluation_gfg/c/`).
28. YARPGen: https://github.com/intel/yarpgen; Csmith: https://github.com/csmith-project/csmith; ldrgen: arXiv 1709.04421.
29. rStar-Coder: arXiv 2505.21297.
30. Jotai: https://github.com/lac-dcc/jotai-benchmarks; DecompileBench: arXiv 2505.11340; Idioms: arXiv 2502.04536;
    Guess & Sketch: arXiv 2309.14396.
