# Evidence: 04-synth (deep-research output, Oct 24, 2026)

Raw synthesis of the 3-vote verified claims. Fields: confidence | claim, SRC (sources), EV (evidence and vote). Sections after the findings: caveats, open questions, refuted claims, sources, stats.

SUMMARY: The PI's differential-oracle idea holds up: the compiled reference gives correct expected outputs for any valid input. The verified evidence shows the two hard parts are input validity and suite adequacy, not the oracle. Pipelines that generate inputs automatically can run most functions: Input-Gen replays about 90% of ComPile functions, while ExeBench gets only 700k executable out of 4.5M compilable C functions (about 15%). But coverage is modest (37% average block coverage with 1 input, 45% with 5). Invalid inputs are common: a third of CodeContests' mutated inputs fail validators, and about 70% of OSS-Fuzz-Gen crashes are harness false positives. So a validity gate that rejects UB and precondition violations must run before any reference output counts as ground truth. Jotai's clang ASan+UBSan no-recover run is a concrete recipe. Weak suites let a lot of wrong code pass: HumanEval+ cuts pass@k by up to 19.3-28.9%; about 18.5% of known-wrong programs pass the CodeContests tests (TACO tests let through over 90% on the hardest problems); the best feedback-refined suite still passes about 9.1% in-sample. False-positive rewards measurably hurt RL. So adequacy has to be measured empirically, as TPR/TNR against pools of known-correct and known-wrong candidates, not assumed. The verified evidence does not cover the statistical-bound part (mutation score with Clopper-Pearson or rule-of-three), symbolic execution, the Csmith/EMI/YARPGen input-count results, test-solution co-evolution loops, or BDD/Given-When-Then specs.

- high | Execution-ready yield from real-world C depends heavily on the method. ExeBench's type-based plus LLM IO generation makes about 15% of compilable GitHub functions executable with IO examples (700k of 4.5M). Input-Gen is compiler-integrated: it works on LLVM IR and generates the initial memory state as well as the arguments. It ends successfully for about 92% of functions and replays about 90%, about 21.5M functions under a 5 s per-input limit. Jotai packages AnghaBench functions as single-file C benchmarks (function, input drivers, main, RNG), split into anghaLeaves (no calls) and anghaMath (calls math.h). The headline numbers use different definitions of 'valid' and are not directly comparable.
  SRC: https://pldi22.sigplan.org/details/maps-2022-papers/6/ExeBench-An-ML-scale-dataset-of-executable-C-functions https://arxiv.org/abs/2406.08843v1 https://github.com/lac-dcc/jotai-benchmarks
  EV: ExeBench (MAPS@PLDI 2022): '4.5M compilable and 700k executable C functions'. It names external types/functions and scalable IO generation as the two blockers. The 15% is derived from these counts. In the current HF release (v1.01) the ratio is about 19-23%. Input-Gen (arXiv 2406.08843, preprint) ran on about a quarter of ComPile (23.75M functions instrumented): 21,880,471 generated (92%), 21,472,918 replayed (90%). Here 'valid' means it runs without crashing under their runtime, not UB-free or precondition-respecting. Jotai: each benchmark has at least input 0, often only a few inputs. A claim that Input-Gen reported ExeBench at about 15% yield and a 6x advantage was refuted 0-3.

- medium | Random inputs give modest coverage, even with compiler guidance. One generated input averages 37% block coverage and five guided inputs reach 45%. Branch hints add about 1 point for most languages (12 for Swift). A suite of a few random inputs leaves most blocks unexercised, so passing it is weak evidence of equivalence unless more inputs are added or another method supplements them.
  SRC: https://arxiv.org/abs/2406.08843v1
  EV: Input-Gen Section 5 and conclusion: 'a single generated input results in an average block coverage of 37%, whereas guided generation of five inputs improves it to 45%.' This is a single preprint source, measured on multi-language ComPile (C, C++, Rust, Swift, Julia) with unstubbed syscalls and external calls. It never tested more than 5 inputs. Small leaf C functions (AnghaBench/ExeBench style) may reach much higher coverage. 'Weak as equivalence evidence' is our inference; the paper does not discuss equivalence.

- high | Invalid inputs are common, so a validity/UB gate has to run before an observed reference behavior counts as ground truth. Jotai's filter is concrete and reusable: keep an input only if it runs to termination under `clang -g -O1 -fsanitize=address,undefined,signed-integer-overflow -fno-sanitize-recover=all` with no leaks. Inputs that fail are dropped, and functions with no passing input are dropped. Precondition violations dominate failures elsewhere too: about a third of CodeContests' mutation-generated inputs fail constraint validators, and instant crashes of LLM-written fuzz harnesses usually come from API misuse, not real bugs.
  SRC: https://github.com/lac-dcc/jotai-benchmarks https://google.github.io/oss-fuzz/research/llms/target_generation/ https://arxiv.org/html/2506.05817
  EV: Jotai README ('Zen of Jotai': well-defined, clean) and the kotai code (CompileBenchmarks.py, _runWithFsanitize) apply the sanitizer filter per input. Kcc and valgrind steps are commented out, and MSan is not used, so uninitialized reads slip through. OSS-Fuzz docs: instant crashes mean 'some miscalled API and the crashes are likely to be false positives'. The verifier found FalseCrashReducer (arXiv 2510.02185), which measures about 70% false-positive crashes (3,390/4,835) caused by bad state/input initialization. CodeContests+: 'Only 0.79 million [of 1.18M] passed validation, which is 67.1%'. Those validators are LLM-written testlib programs with unmeasured accuracy, from one dataset. A claim that OSS-Fuzz-Gen harnesses succeeded on 14/31 projects was refuted 0-3.

- low | Preconditions inferred by an LLM cannot be trusted just because they prove. LLM-generated ACSL annotations can be syntactically valid and verifiable yet semantically weak (trivially true or overly conservative). A pipeline that uses LLM-inferred contracts to constrain input generation needs a separate strength check, such as mutation-based completeness.
  SRC: https://arxiv.org/abs/2602.13851
  EV: Beg, O'Donoghue and Monahan (arXiv 2602.13851 v3): 355 CASP C programs, Frama-C WP, three LLMs plus rule-based and RTE baselines. 'Some annotations—especially those produced by LLM-based methods—are syntactically valid but semantically weak ... verification success alone does not fully capture annotation usefulness.' The observation is qualitative only: no counts, rates or precondition-specific numbers. The original wording 'often' overstates the paper's 'some'. The verifier found related work (Dafny Mutation-CEGIS; Mutation-based Completeness Rate, arXiv 2503.04779; SpecRL, arXiv 2604.05820) that uses mutants or tests to catch vacuous specs. Those papers were not themselves verified here.

- high | Standard benchmark and RL test suites let a large share of wrong programs pass, and hard problems are the worst case. HumanEval+ (80x more tests) cuts pass@k by up to 19.3-28.9%. On Codeforces difficulty-4 problems (Qwen2.5-Coder-7B programs), TACO precision is 9.82 and CodeContests 6.67, so more than 90% of accepted programs are wrong. Against a pool of about 1.1e7 solutions, about 18.5% of known-wrong solutions pass the CodeContests tests (TNR 81.52). Even the best feedback-refined suite lets about 9.1% through, and that figure is in-sample.
  SRC: https://nips.cc/virtual/2023/poster/72990 https://arxiv.org/html/2505.24098v1 https://arxiv.org/abs/2601.13682
  EV: EvalPlus (NeurIPS 2023): 'catch significant amounts of previously undetected wrong code ... reducing the pass@k by up-to 19.3-28.9%'. This is a lower bound on false passes, and the range is a maximum across k. HardTests (arXiv 2505.24098, NeurIPS 2025 / ICLR 2026): 'false positive rate being more than 90% for difficult problems'. The ICLR camera-ready drops that sentence and reports TACO 17.83 vs HardTests 55.88 precision on combined difficulty 4+. The gap shrinks for stronger program sources (GPT-4o 62.64 vs 79.82). About 30% of incorrect LLM programs are TLE failures, which matters less for transpilation. CodeContests-O Table 2 (arXiv 2601.13682, preprint): CodeContests TPR/TNR 85.18/81.52. The pool excludes solutions that fail to compile or fail the public tests. A refuted claim (APPS 60% false-positive, 1-2) is excluded.

- medium | How the inputs are generated matters more than how many there are. Inputs written directly by an LLM give the weakest verifier. Programs that generate random inputs within the stated constraints, plus adversarial 'hacking' generators aimed at likely wrong solutions, both behind an LLM-written input validator, give the best precision and recall. Iterating with feedback against known-wrong solutions also helps. Adding raw test density raises TNR but lowers TPR.
  SRC: https://arxiv.org/html/2505.24098v1 https://iclr.cc/media/iclr-2026/Slides/10006843.pdf https://nips.cc/virtual/2023/poster/72990 https://arxiv.org/abs/2601.13682
  EV: HardTests poster (Qwen2.5-Coder-7B, avg / Diff 4+ precision): LLMGen-only 42.4/8.0; CodeContests 51.6/10.8; TACO 61.0/17.8; LLMGen+random/special generators 64.7/47.7; full pipeline with HackGen 79.2/55.9. Paper average gains: +11.3 precision and +17.5 recall. These may be computed against TACO only; the ICLR camera-ready says +11.22. LLM-direct inputs score lowest on precision but highest on recall (95.90 in v1), so they are lenient, not strict. EvalPlus seeds inputs with ChatGPT and then applies type-aware mutation (9.6 to 764.1 tests per problem). HumanEval+-mini is about 47x smaller with similar effectiveness, so test count is not adequacy. CodeContests-O: up to 3 refinement rounds against S+/S- pools reach TPR/TNR 89.37/90.89. CodeContests+ 1x to 5x: TNR 86.03 to 89.35, TPR 85.68 to 83.84. HardTests: 68.06/89.17. Two of these claims passed 2-1.

- high | Suite adequacy can and should be measured empirically, as binary classification against pools of known-correct and known-incorrect programs. TPR (correct code accepted) measures test correctness. TNR (wrong code rejected) measures coverage. This is the direct template for estimating a false-pass rate. For transpilation the negative pool would have to be built from wrong model translations and mutants of the reference (our inference; no verified source does this for assembly).
  SRC: https://arxiv.org/html/2506.05817 https://arxiv.org/abs/2601.13682 https://arxiv.org/html/2505.24098v1
  EV: CodeContests+ Section 4.4: up to 100 correct and 100 incorrect submissions per problem, keeping problems with at least 10 of each. That gives 10,166 problems, 1.72M submissions, about 200 tests each, and more than 300M executions. CodeContests-O evaluates against about 1.1e7 verified solutions, but the same S+/S- pools drive its refinement, so its 9.1% false-pass rate is optimistic and in-sample. HardTests uses real Codeforces judge verdicts (subsampled) and official AtCoder tests as ground truth. The paper's 'FPR' means 1 - precision. Under a compiled-reference oracle, expected outputs are correct by construction. TPR loss would then come only from invalid inputs or behavior that differs between targets, so TNR becomes the main quantity to estimate (our inference).

- medium | Verifier quality changes RL outcomes, and false-positive rewards are the dangerous kind. In GRPO on Qwen3-4B with a binary all-tests-pass reward, HardTests raised LiveCodeBench-105 pass@10 from 56.19 to 64.76. TACO tests on the same about 5k problems gave 57.14 and dropped pass@1 below base (38.48 to 36.95). In AceReason-7B, mixing in problems whose weak tests let wrong solutions pass caused a sudden mid-training drop in held-out accuracy, not just slower learning. The authors attribute this to the policy learning to skip corner cases or to emit brute-force solutions.
  SRC: https://arxiv.org/html/2505.24098v1 https://iclr.cc/media/iclr-2026/Slides/10006843.pdf https://arxiv.org/pdf/2505.16400
  EV: HardTests Table 5 (pass@1/5/10): base 38.48/52.04/56.19, TACO 36.95/51.01/57.14, HardTests 39.42/57.89/64.76. Each arm is a single run with no seeds or CIs, on 105 eval problems, with the checkpoint chosen by validation reward. The robust result is HardTests over TACO (+7.6 pass@10). 'TACO hurts' (-1.5 pp pass@1) is probably within noise, and TACO actually raised pass@10. The poster's 'bad rewards are worse than no RL' is the authors' framing. In distillation, scale (46.6k unfiltered vs 13k filtered) beat correctness, but scale and filtering are confounded. AceReason-Nemotron (arXiv 2505.16400v3, Section 4.3.6, Fig. 5): a single ablation with an unstated mixing ratio and no false-positive rate measured. The false-negative arm plateaus; the false-positive arm drops sharply.

== caveats
"1. Domain mismatch. Most measured false-pass rates come from competitive programming (stdin/stdout Python/C++), not C functions with pointer and memory state, and not assembly. Part of the false-pass rate there is TLE/inefficiency, which matters less for checking transpilation equivalence. Comparing function outputs will also need the final memory state of pointer arguments, not just return values (our inference). Behavior that differs between targets, such as char signedness on x86 versus AArch64 or uninitialized reads that ASan/UBSan miss, could make the oracle itself target-dependent. No verified source addressed this.\n2. Statistical weakness. The RL results (HardTests vs TACO, AceReason) are single runs with no seeds or error bars. CodeContests-O's 9.1% is in-sample. HardTests' 11.3/17.5-point averages are unclear in construction and changed in the camera-ready (+11.22; the '>90%' sentence was removed). The difficulty-4 precision of 50.00 likely rests on a small sample.\n3. Source quality. Input-Gen, CodeContests+, CodeContests-O and the ACSL paper are un-refereed preprints. The ACSL weak-spec finding is qualitative with no rates. The CodeContests+ validators are LLM-written with unmeasured accuracy. Several verifiers could not run contradiction searches because the shared web-search budget ran out.\n4. Coverage gaps. No verified claim covers: symbolic or concolic execution (KLEE, SymCC); property-based testing; Daikon-style precondition inference; MSan, CompCert or Cerberus; Csmith, EMI, YARPGen or alive2 results on how many random inputs detect miscompilations (research question 3 is entirely unanswered); mutation score with Clopper-Pearson, rule-of-three or capture-recapture bounds (the statistical half of research question 2); STaR, CURE, CodeT or ReasonFlux co-evolution loops and test collapse (research question 4 is covered only by the CodeContests-O feedback loop and the AceReason failure mode); and BDD/Given-When-Then specification (research question 5). Verification was also cut short by the session limit: 31 votes failed for this topic, so some of these gaps may be unverified claims, not missing evidence.\n5. Refuted claims: the Input-Gen-vs-ExeBench 6x comparison (0-3), the OSS-Fuzz-Gen 14/31 harness success rate (0-3), Jotai's BigArray/DSL description (1-2), CodeContests+'s Generator/Validator loop details (1-2), and the APPS 60% / CodeContests 46%-TLE figures (1-2).\n6. Time-sensitivity. Sources run from 2022 to April 2026. ExeBench's current HF release counts differ from the paper's (compilable-to-executable ratio about 19-23% vs 15%).\n7. Project relevance. Nothing here yet measures a synthetic suite's false-pass rate for C-to-assembly transpilation. The evidence therefore supports keeping the current real-I/O-only training-data rule until such a suite's TNR has been measured against a negative pool."
== openQuestions
[
 "How many sanitizer-validated, constraint-following random inputs per function does a small AnghaBench/ExeBench-style C function need before the 95% Clopper-Pearson upper bound on the false-pass rate falls below a target (e.g., 1%)? The negative pool here would be wrong model translations plus assembly-level mutants of the reference. Do symbolic/concolic inputs (KLEE/SymCC) or adversarial LLM 'hacking' generators lower that count?",
 "Does a feedback loop that refines tests against known-wrong translations (CodeContests-O style) hold its TNR on held-out wrong translations from a later policy checkpoint? What safeguards prevent tests from overfitting to the current negative pool, or collapsing toward easy tests, when the pipeline grows the data iteratively?",
 "How often does a signature-driven generator with an ASan+UBSan (no MSan) gate still let through invalid or target-dependent inputs (uninitialized reads, char signedness, implementation-defined shifts)? Those inputs would make the x86 reference a wrong oracle for ARM/RISC-V. What is the gate's false-rejection rate on functions whose preconditions are unknown?",
 "Do Csmith/EMI/YARPGen/alive2 or binary differential-testing studies give usable priors on how many random inputs detect a semantic difference between two compiled versions of the same function? Can LLM-inferred ACSL preconditions be checked for strength (e.g., by mutation completeness) well enough to constrain input generation?"
]
== refuted
[
 {
  "claim": "Input-Gen reports that the ExeBench IO-generation method yields runnable input-output pairs for only about 15% of the C functions it processes. If true, a runtime that lazily materializes memory could make about 6x more functions executable than ExeBench-style type-based synthesis.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2406.08843v1"
 },
 {
  "claim": "Inputs come from named constraint sets (for example big-arr, big-arr-10x) written in a domain-specific language. The 'BigArray' set gives every pointer argument a memory region as large as the largest integer argument, which is a heuristic for inferring buffer-size preconditions from the signature. The README does not document the DSL; a report is said to be in preparation.",
  "vote": "1-2",
  "source": "https://github.com/lac-dcc/jotai-benchmarks"
 },
 {
  "claim": "In the initial OSS-Fuzz-Gen evaluation, LLM-generated fuzz harnesses for C/C++ library functions compiled and raised coverage in fewer than half of the projects tried: 14 of 31 OSS-Fuzz projects. Generating valid harnesses or inputs from function signatures with an LLM alone therefore fails often, and per-benchmark build rates ran from about 6% to 87.5%.",
  "vote": "0-3",
  "source": "https://google.github.io/oss-fuzz/research/llms/target_generation/"
 },
 {
  "claim": "The test-generation pipeline is an LLM Generator agent that writes input-generator programs, paired with a Validator that checks each input against the problem's constraints and sends errors back to the Generator until every input is valid. Validator bugs are caught by running the validator on the problem's sample inputs. The authors admit that a constraint the LLM identified wrongly is not caught automatically.",
  "vote": "1-2",
  "source": "https://arxiv.org/html/2506.05817"
 },
 {
  "claim": "Existing code-RL test suites have high measured false-positive rates: on APPS, 60% of programs that pass the provided tests are wrong, and on CodeContests, 46% of passing programs are too slow (the tests miss worst-case or TLE inputs).",
  "vote": "1-2",
  "source": "https://iclr.cc/media/iclr-2026/Slides/10006843.pdf"
 }
]
== unverified
[]
== sources
[
 {
  "url": "https://arxiv.org/abs/2406.08843v1",
  "quality": "primary",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 5
 },
 {
  "url": "https://homepages.dcc.ufmg.br/%7Efernando/publications/papers/CondeKindCOLA25.pdf",
  "quality": "unreliable",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 0
 },
 {
  "url": "https://pldi22.sigplan.org/details/maps-2022-papers/6/ExeBench-An-ML-scale-dataset-of-executable-C-functions",
  "quality": "primary",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 5
 },
 {
  "url": "https://github.com/lac-dcc/jotai-benchmarks",
  "quality": "primary",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 5
 },
 {
  "url": "https://google.github.io/oss-fuzz/research/llms/target_generation/",
  "quality": "primary",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2602.13851",
  "quality": "primary",
  "angle": "Valid input generation from C signatures (preconditions, UB rejection)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2505.24098v1",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2506.05817",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 5
 },
 {
  "url": "https://nips.cc/virtual/2023/poster/72990",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 4
 },
 {
  "url": "https://arxiv.org/abs/2601.13682",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 5
 },
 {
  "url": "https://iclr.cc/media/iclr-2026/Slides/10006843.pdf",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2505.16400",
  "quality": "primary",
  "angle": "Measured false-positive rates of test suites (code RL benchmarks)",
  "claimCount": 5
 },
 {
  "url": "https://people.cs.umass.edu/~rjust/publ/mutants_real_faults_fse_2014.pdf",
  "quality": "primary",
  "angle": "Mutation analysis and statistical bounds on test adequacy",
  "claimCount": 5
 },
 {
  "url": "https://conf.researchr.org/details/icse-2018/icse-2018-Technical-Papers/78/Are-Mutation-Scores-Correlated-with-Real-Fault-Detection-A-Large-Scale-Empirical-stu",
  "quality": "primary",
  "angle": "Mutation analysis and statistical bounds on test adequacy",
  "claimCount": 4
 },
 {
  "url": "https://www.cefns.nau.edu/~adg326/issre15.pdf",
  "quality": "primary",
  "angle": "Mutation analysis and statistical bounds on test adequacy",
  "claimCount": 5
 },
 {
  "url": "https://discovery-pp.ucl.ac.uk/id/eprint/1499169/1/Jia_Trivial_Compiler_mutation-testing-papadakis-icse15.pdf",
  "quality": "primary",
  "angle": "Mutation analysis and statistical bounds on test adequacy",
  "claimCount": 5
 },
 {
  "url": "https://ntrs.nasa.gov/citations/19930007588",
  "quality": "secondary",
  "angle": "Mutation analysis and statistical bounds on test adequacy",
  "claimCount": 5
 },
 {
  "url": "https://www2.cs.utah.edu/~regehr/papers/arm-tv.pdf",
  "quality": "primary",
  "angle": "Differential testing and translation validation for compilers/binaries",
  "claimCount": 5
 },
 {
  "url": "https://www-old.cs.utah.edu/~regehr/alive2-pldi21.pdf",
  "quality": "primary",
  "angle": "Differential testing and translation validation for compilers/binaries",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2109.13498",
  "quality": "primary",
  "angle": "Differential testing and translation validation for compilers/binaries",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2309.14396",
  "quality": "primary",
  "angle": "Differential testing and translation validation for compilers/binaries",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.03136",
  "quality": "primary",
  "angle": "Self-improving test/solution co-evolution loops and their failure modes",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.24098",
  "quality": "primary",
  "angle": "Self-improving test/solution co-evolution loops and their failure modes",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2502.14948",
  "quality": "primary",
  "angle": "Self-improving test/solution co-evolution loops and their failure modes",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2502.01619",
  "quality": "primary",
  "angle": "Self-improving test/solution co-evolution loops and their failure modes",
  "claimCount": 5
 }
]
== stats
{
 "angles": 5,
 "sourcesFetched": 25,
 "claimsExtracted": 118,
 "claimsVerified": 25,
 "confirmed": 20,
 "killed": 5,
 "unverified": 0,
 "afterSynthesis": 8,
 "urlDupes": 0,
 "budgetDropped": 5,
 "agentCalls": 107
}
