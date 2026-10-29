# 04. Synthetic tests with statistical guarantees (the separate arm)

**Status (Oct 24, 2026): design.** Tags: [V] verified, [R] repo, [L] unverified literature, [P] proposal.
Scope decision (Oct 24): synthetic tests are in scope **as a separate, labeled arm**. The main corpus stays real-test
([01](01-data.md)). No teacher models: only the compiler, program-analysis tools and the model itself.

## Summary
- **The PI's core idea holds.** In translation, the compiled reference is a correct output oracle for every valid input
  [V]. Expected outputs are therefore free. The two hard parts are **input validity** and **suite adequacy**:
  - **Validity:** about a third of mutation-generated CodeContests inputs fail constraint validators, and about 70% of
    LLM-harness fuzz crashes are false positives [V]. Every input has to pass a validity gate before its reference output
    counts as ground truth.
  - **Adequacy:** typical suites let 9–18% of known-wrong programs through, and on hard problems far more [V].
    Coverage from random inputs is modest: 37% of blocks with 1 input, 45% with 5 [V]. Adequacy has to be **measured**
    against pools of known-wrong programs, not assumed [V].
- **Bad tests hurt RL.** RL on weak tests ended at or below RL on strong tests, and mixing in false-positive problems
  caused a sudden mid-training collapse [V]. The pipeline's output therefore goes into training only when its estimated
  false-pass rate is below a pre-registered bound.
- **The design, "Given/When/Then" built by program analysis:**
  - **Given:** a typed program state, derived from the signature and from inferred preconditions.
  - **When:** the call.
  - **Then:** the reference's full observable effect.
  - All three are generated programmatically, never by an LLM. LLM-written C contracts are often vacuous [V, low].
- **The statistical core is calibration on tasks that also have real tests.** On those we can measure the suite's
  false-pass rate with exact or bootstrap bounds, learn how per-task adequacy maps to false passes, and only then apply
  the pipeline to the 1M+ functions that have no tests.

## 1. Pipeline
```
C function ─► 1. compile for all targets ─► 2. GIVEN: typed state model ─► 3. WHEN: input generation
  (AnghaBench,      (x86, ARMv8-macOS,          (shapes, sizes, preconditions)   (boundary + random + coverage-guided
   ExeBench C,       AArch64-Linux, RISC-V)                                        fuzzing + symbolic)
   Decompile-Bench)                                                                       │
                                                                                          ▼
  7. suite minimization ◄── 6. adequacy (coverage, mutants) ◄── 5. THEN: observe ◄── 4. validity gate
     (set cover, ≤ k tests)      per task                       (return, memory,        (sanitizers, termination,
                                                                  globals, stdout)        determinism, cross-ISA agreement)
```

### 1.1 Compile (stage 1)
- Build each function standalone for x86-64 and every target at O0 and O2 with the pinned toolchains ([02](02-harness.md)).
- Keep functions whose callees are absent, or resolvable within the file, or in libc/libm.
- Jotai already packages AnghaBench functions this way, as single-file benchmarks with drivers. Its split into leaves
  (`anghaLeaves`) and math-calling functions (`anghaMath`) is a ready starting pool [V].

### 1.2 GIVEN: typed state model (stage 2) [P]
- **Parse with libclang** to get parameter types, struct layouts and the globals the function reads or writes.
- **Shapes:**
  - a scalar carries a range from its type;
  - `T*` is a buffer of length n_T, with nullability tried both ways;
  - `char*` is a buffer that has a NUL within its length;
  - a struct pointer is a recursive shape with a depth bound;
  - function pointers reject the function in v1.
- **Preconditions without an LLM:**
  1. **Static bounds:** relations between the trip counts of indexing loops and integer parameters (for example
     `for (i < n) a[i]` gives |a| ≥ n), from a simple dataflow pass.
  2. **Dynamic sizing:** start with large buffers under ASan redzones, find the furthest access and shrink to the minimal
     safe size.
  3. **Daikon-style relations:** infer them from successful runs and enforce them on new inputs [L].
- Write each test as data:
  `{given: {args, buffers, globals}, when: {call}, then: {return, buffers_after, globals_after, stdout, exit}}`.
  One driver generator per target turns it into C. This is the "Given/When/Then" form, kept strict and machine-checkable.

### 1.3 WHEN: input generation (stage 3)
- Generators are combined and ablated one at a time (S-2):
  1. **Boundary values per shape:** 0, ±1, the type's minimum and maximum, empty and one-element buffers, n at the
     inferred bound.
  2. **Type-aware random** inputs within the shapes (ExeBench-style [V]).
  3. **Coverage-guided fuzzing** of the reference with a structure-aware mutator over the shapes, using libFuzzer with
     coverage [L].
  4. **Symbolic/concolic** inputs for branches still uncovered (KLEE/SymCC) [L].
  5. **The policy itself, optionally, as an ablation:** the model proposes inputs in the Given format. This fits the
     "model itself" rule.
- Input-Gen, a compiler-integrated generator that also builds initial memory, replays about 90% of functions [V]. That is
  the yield benchmark to compare against.

### 1.4 Validity gate (stage 4)
An input is kept only if all of these hold:
1. **No UB:** it runs to termination under `clang -g -O1 -fsanitize=address,undefined,signed-integer-overflow
   -fno-sanitize-recover=all` with no leaks. This is the Jotai filter [V]. Also run MSan on Linux, because ASan and
   UBSan miss uninitialized reads [V caveat].
2. **Termination** within a CPU budget, and **determinism**: two runs with different ASLR give identical observations.
3. **Cross-ISA agreement [P]:** the reference built for x86-64, ARMv8-macOS, AArch64-Linux and RISC-V64 gives identical
   observations. This removes implementation-defined behavior, which would otherwise make the x86 reference a wrong
   oracle for the target. Examples are `char` signedness (signed on Apple arm64, unsigned on Linux AArch64 [L]),
   oversized shifts, alignment, and uninitialized reads the sanitizers miss.
   Functions with no input that passes all gates are dropped.

### 1.5 THEN: observation (stage 5)
- Record the return value, the final bytes of **every buffer in the Given state**, the globals touched, stdout and stderr,
  and the exit status.
- Comparing only return values would let wrong side effects pass. Forklift checks return values plus side effects on
  arrays, struct pointers and globals [V].

### 1.6 Adequacy (stage 6), measured per task
- **Coverage:** branch and PC coverage on the reference target assembly. GG reports more than 98% PC coverage on its
  benchmarks [R paper].
- **Mutation score:**
  - **Assembly-level mutants** of the reference target asm, with operators copied from the errors GG observed:
    immediates off by one, stack or array offsets changed, register swaps, flipped condition codes, wrong load or store
    width, a dropped instruction, swapped operands (paper Table 8) [R].
  - **Plus C-level mutants**, compiled.
  - **Equivalent mutants** are removed when they survive an audit suite of about 10k valid inputs, and, where it
    terminates, an angr/z3 check, which D-LiFT reports gives a result for about 80% of functions [V].
- **Natural negatives,** on calibration tasks only: policy translations that fail the real tests.

### 1.7 Minimization (stage 7)
- Greedy set cover over covered branches ∪ killed mutants, keeping at most k = 20 tests per task, to bound reward
  cost ([02](02-harness.md)).
- Keep the full suite for audits.

## 2. Statistical validation (the paper's core)
### 2.1 Estimands
- **FPR (false-pass rate):** P(a translation passes the suite | it is wrong). This is the one that hurts RL [V].
- **FNR (false-fail rate):** P(it fails the suite | it is correct). It should be close to 0 by construction; a nonzero
  value points to a validity-gate leak.
- **"Wrong" and "correct"** are decided by a **gold oracle**:
  - fails the task's real tests, **or**
  - fails a frozen **audit suite** (about 10k gated inputs plus symbolic checks where they terminate).

  The audit suite is never used to train or to refine tests.

### 2.2 Calibration set
- Tasks that have real tests and are function-level: MBPP-C (952) and HumanEval-C (164).
  - HumanEval-C serves only as P2's final held-out calibration stratum and never as a training reward.
  - Add DecompileBench functions with seed-corpus profiles as a third, real-code stratum [V].
- Run the pipeline on these tasks **without looking at their real tests**.

### 2.3 Negative pool
- For each calibration task, sample about 32 translations from each of several policies: base GG, SFT variants and RL
  checkpoints, at T = 1. Label every translation with the gold oracle.
- Wrong translations are natural negatives. Mutants are reported **separately**: mutation score is a proxy, and some real
  faults are not coupled to any mutant [L; a claim of 27% failed verification at the session limit].

### 2.4 Estimators [P]
- **Pooled FPR:** the share of escaped wrong translations, with a 95% **cluster bootstrap** over tasks, because wrong
  translations cluster within tasks.
- **Per stratum, Clopper–Pearson.** If no wrong translation escapes among n, the one-sided 95% upper bound is about
  3/n (rule of three). Claiming FPR < 1% therefore needs about 300 effectively independent wrong translations with no
  escapes. Report the **number of tasks with any escape** next to the pooled rate.
- **FNR:** from the references plus passing natural translations, with the same estimators.
- **Calibrated adequacy:**
  - Fit P(escape | per-task coverage, mutation score, number of tests, shape features) on calibration tasks.
  - Check the calibration on a held-out stratum with a reliability diagram and the Brier score.
  - Apply the model to the test-free pool to **predict pool-level FPR with a CI**.
  - This is the bridge from "measured where we have gold" to "estimated where we don't". Distribution shift between
    MBPP-style and AnghaBench functions is the main threat, so report it per stratum (pointer args, loops, structs, size).
- **The audit applies to our current reward too.** Measure the FPR of MBPP-C's own asserts (median 3 per task [R])
  against the audit suite. This is EvalPlus's question, which cut pass@k by up to 19.3–28.9% on HumanEval [V]. If MBPP-C's
  real tests let wrong translations pass, every RL result so far is affected. That makes this a P0 measurement for P1.

### 2.5 Gate for using synthetic tasks in RL (pre-registered) [P]
- A task enters the synthetic pool if its mutation score is at least τ and its branch coverage at least κ (fixed on
  calibration data).
- The pool is used only if the **upper 95% bound on the predicted pool FPR is ≤ 2%** and the measured FNR is ≤ 0.5%.
  For scale: weak suites that let 9–18% through hurt RL [V].

## 3. Growing the data iteratively
Round r = 0, 1, 2, 3 [P]:
1. Train π_r on R ∪ S_r: real tasks plus the synthetic pool, compute-matched against the R-only arm.
2. Sample π_r on the synthetic pool. For any translation that passes its suite but fails the **audit** suite, add the
   distinguishing input to that task's suite. This is counterexample-guided refinement, in the style of CodeContests-O,
   whose in-sample TPR/TNR rose to 89.4/90.9 [V].
3. **Re-estimate FPR on held-out negatives from the next checkpoint π_{r+1}.** CodeContests-O's 9.1% false-pass rate is
   in-sample and optimistic [V], so the estimate must use negatives the refinement never saw.
4. Expand the pool with new functions that pass the gate, and repeat.

**Stop when** the held-out FPR stops falling, or when gains on the tier-R evals stop. Tier-R evals stay the
arbiter of whether synthetic data helps.

**Safeguards:**
- The audit suite and the calibration hold-out are frozen.
- Track the gap between suite pass rate and audit pass rate during training. A widening gap means the policy is
  exploiting the suites.
- Never refine on eval tasks.

## 4. Experiments (detailed in [06](06-schedule.md), track S)
| ID | Question | Output |
|---|---|---|
| S-0 | Yield funnel: compiled → shaped → ≥ 1 valid input → coverage ≥ κ → mutation score ≥ τ, per source | counts per stage, compared with ExeBench 15% and Input-Gen 90% [V] |
| S-1 | How does FPR fall with the number of inputs (1…1,000), per generator? | adequacy curves with CIs; the k at which the upper bound reaches ≤ 2% |
| S-2 | Which generator components matter, and what does each gate cost in FNR? | component ablation table |
| S-3 | What is the FPR of the real tests we use (MBPP-C, HumanEval-C asserts)? | real-test audit; feeds into P1 |
| S-4 | Does synthetic RL data help on **real-test** evals? | R-only vs S-only vs R+S, compute-matched, 3 seeds |
| S-5 | How does the effect scale with pool size (1k → 10k → 100k)? | ScaleRL fits per arm |
| S-6 | Does iteration hold FPR down on held-out later negatives? | FPR per round |
| S-7 | Does the policy learn to hack synthetic suites? | suite-vs-audit gap over training |

## 5. Risks, stated plainly
- **Function-level real-test evals are small** (HumanEval-C 164, MBPP dev 94). S-4's detectable effect is about 7–9 pp
  ([05](05-statistics.md)). P2's strongest claims will be about the **pipeline's measured validity** (S-0 to S-3, S-6).
  RL utility is a secondary claim, unless a larger function-level real-test set is built, for example from DecompileBench
  functions.
- **Distribution shift** from calibration tasks to AnghaBench: handled per stratum, never fully removable.
- **AnghaBench is in GG's SFT data** [R]. RL on it teaches less than RL on new code. Prefer the non-AnghaBench sources
  for the synthetic RL pool, and use AnghaBench for the yield and validity studies.
- **Gaps in the evidence base:** symbolic execution, property-based testing, and Csmith/EMI/YARPGen-style input-count
  results were not covered by verified claims. That part of the design rests on [L] and has to be measured by us.

## Sources
Verified claims: [evidence/04-synth.md](evidence/04-synth.md).
- ExeBench: MAPS 2022 (pldi22.sigplan.org); Input-Gen: arXiv 2406.08843; Jotai: https://github.com/lac-dcc/jotai-benchmarks
- OSS-Fuzz-Gen false positives: https://google.github.io/oss-fuzz/research/llms/target_generation/; FalseCrashReducer: arXiv 2510.02185
- CodeContests+: arXiv 2506.05817; CodeContests-O: arXiv 2601.13682; HardTests: arXiv 2505.24098; EvalPlus: NeurIPS 2023
- AceReason-Nemotron: arXiv 2505.16400; LLM-generated ACSL: arXiv 2602.13851; Forklift: arXiv 2404.16041; D-LiFT: arXiv 2506.10125
