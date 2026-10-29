# Paper plan: top-down design for SFT + RLVR on GG (x86 → ARMv8 / AArch64 / RISC-V)

**Status (Oct 24, 2026): research done, plan proposed, nothing implemented.** It was written for Enzo and for a pitch to
the GG authors. **Design comes first, without feasibility limits** (Enzo, Oct 24). Which parts to run, and on what
resources, is decided later. Licenses are deferred until then too.

## Files
| File | Content |
|---|---|
| [01-data.md](01-data.md) | every source we can train and evaluate on, its oracle tier, counts, licenses, and the new GG-XL eval |
| [02-harness.md](02-harness.md) | training stack, reward fleet, targets and ABIs, verification modes, anti-hacking, efficiency measurement |
| [03-curriculum.md](03-curriculum.md) | difficulty-aware scheduling: what works, what doesn't, and our layers L0–L4 |
| [04-synthetic-tests.md](04-synthetic-tests.md) | the Given/When/Then differential-test pipeline and its statistical validation (the separate arm) |
| [05-statistics.md](05-statistics.md) | endpoints, tests, power, multiplicity, seeds, pre-registration |
| [06-schedule.md](06-schedule.md) | phases A–E, tracks S and X, every value with its reason, sweeps, gates, compute |
| [evidence/](evidence/) | the 6 deep-research outputs: verified claims with quotes, votes, refuted claims, open questions |

## Evidence tags
- **[V]** passed this run's adversarial check (3 votes; at least 2/3 needed to refute). The source and quote are in
  `evidence/`.
- **[R]** measured in this repo; the path is given.
- **[L]** literature or general knowledge **not** verified in this run. Check it before citing.
- **[P]** our proposal or inference.

**Coverage.** 6 topics, 143 sources fetched, about 700 claims extracted. A budget of 25 claims per topic went to
verification: 125 confirmed and 25 refuted. Many named methods have no verified claim, which is not evidence against
them. The gaps are listed at the end of each file.

## 1. What we propose
**One shared pipeline, two papers, with rigor first.** The split matters less than the rigor (Enzo, Oct 24). The
infrastructure is built once and serves both.

| Paper | Claim | Venue type | Core experiments |
|---|---|---|---|
| **P1: Recipe** | A rigorous RLVR recipe for assembly transpilation: SFT redo at about 1.5B, real-test RL data at scale, curriculum that targets wasted and zero-pass prompts, CoT arms, several ISAs, judged on a powered held-out benchmark | ML (ICLR / NeurIPS / ICML) | B, C, D, X-1/2, E |
| **P2: Tests** | Differential Given/When/Then test synthesis for binary translation, with **calibrated false-pass bounds**, an audit of existing benchmark tests, and RL utility | SE/PL (ICSE / FSE / ISSTA / OOPSLA) or ML Datasets & Benchmarks | S-0 to S-7, A-6 |
| (P3, optional) | GG-XL benchmark and contamination audit as a resource | NeurIPS Datasets & Benchmarks | A-2, A-3 |

The efficiency objective (X-3) goes into P1 if it works and is ablated there, with λ = 0 as the control.

## 2. Findings that change the plan
1. **RL data with real tests can grow about 50x.** Tests belong to a problem, and each accepted C solution is a
   different x86 prompt. Problems with official tests (Codeforces, AOJ, AtCoder) give about 378k prompts, or about 44k
   at 10 per problem, against 858 today [R].
2. **Our evals can't see the effects RL produces.** The detectable effect is about 7 pp on HumanEval-C, about 9 pp on dev
   and about 12 pp on clean BringUp [V+R]. GG-XL, with at least 1,000 held-out problems, brings it to about 3 pp [P].
3. **57% of run 1's groups had zero gradient** [R]. Filtering (L0) comes first. It saves 2–3x rollouts at matched
   accuracy in math [V].
4. **Run 1's regime was tiny.** It ran 106 steps with the LR decaying to 0 (an average of about 5e-7) and 128 completions
   per step [R]. Published 1.5–8B recipes run 400–4,380 steps at a constant 1e-6 with 512–2,048 completions [V].
   RL-v1 fixes this, and adds FP32 logits and clip-higher [V].
5. **Rewards are the bottleneck, not the GPU.** One 8-worker VM over an SSH tunnel serves everything [R]. Published
   harnesses use a router in front of a worker fleet and score each completion as it finishes [V].
6. **Linux AArch64 is a different target from Apple arm64:** x18, variadic arguments and `va_list` all differ [V].
   GG's ARMv8 numbers use the Apple ABI, so GG comparisons keep a macOS reward target.
7. **Our "real" tests may be weak.** MBPP-C has a median of 3 asserts [R]. Standard suites let 9–18% of wrong programs
   through, and EvalPlus cut pass@k by up to 19–29% [V]. Auditing MBPP-C's asserts (A-6) is a P0 measurement.
8. **Synthetic tests: the oracle is free, but validity and adequacy are not.** The pipeline goes into training only if
   the measured false-pass bound is ≤ 2%. False-positive rewards caused collapses in published code RL [V].
9. **For prompts the model never solves, guidance is the only lever with evidence.** Three groups report it [V]. Our
   source of correct prefixes is the compiler's own target asm. **Splice mode** grades one predicted function inside a
   whole program, linked with the compiler's code for the rest, and gives dense real-test credit [P].
10. **Pick the SFT checkpoint for RL by validation loss and pass@64, not by SFT pass@1** [V]. Better SFT pass@1 can
    lead to worse results after RL [V].

## 3. Why redo SFT at a similar size
1. **The RL result depends on the SFT start.** The best RL start is chosen by validation loss and pass@large-k at the
   SFT plateau [V]. GG's released checkpoint was never chosen for RL. Owning SFT lets us choose it.
2. **Controls we can't run otherwise:**
   - RL problems kept out of SFT, or put in (the data-control arm);
   - eval problems out of SFT. 12 of 65 BringUp programs are in GG's SFT data [R];
   - one multi-target model vs per-target models.
3. **Known SFT levers, measurable only when we own SFT:**
   - context length (4k → 16k gave GG +10 pp; 17% of its BringUp failures were truncation [R paper]);
   - unique data scale (it sets the post-RL ceiling [V]);
   - the tokenizer extension (+1.04 pp [R paper]);
   - a newer base.
4. **Replication.** The released `training_args.bin` differs from the paper text on batch size, optimizer and weight
   decay [R]. A redo confirms the baseline we build on.
5. **Similar size (about 1.5B) keeps the claim about method, not scale.** It stays directly comparable with GG-1.5B, and
   it makes 5-seed confirmatory RL affordable. GG's own results show the size range matters: 0.5B scored 25.5% and 1.5B
   45.1% on HumanEval O2 [R paper].
6. **Base choice by a pre-declared screen** ([06](06-schedule.md) B-0): tokenizer fertility on our asm, native context,
   and validation loss plus pass@64 after identical 200k-pair SFT. Qwen2.5-Coder-1.5B-Instruct is always kept as the GG
   anchor.

## 4. Where the evidence qualifies the starting ideas (honest disagreements)
- **"Curriculum by program difficulty."**
  - Static difficulty is useful as a **prior** and to define **stages**. As a fixed easy-to-hard order it often loses
    to random sampling [V].
  - The working signal is the policy's own pass rate.
  - Curriculum mostly buys **efficiency**, not a higher ceiling [V]. The levers that raise the ceiling are data scale,
    SFT quality, batch size and precision [V], so the schedule puts those first.
- **"Given/When/Then unit tests with really good numeric results."**
  - Given/When/Then is a good **representation**. The quality comes from the validity gate, observing the full state, and
    calibration against tasks with real tests.
  - With no teacher model, every part is generated by program analysis and fuzzing. LLM-written contracts are often
    vacuous anyway [V].
- **"Data guaranteed to be good."**
  - Even official tests can be weak, and the 3-assert MBPP-C tests are the case at hand.
  - "Guaranteed" has to be measured (A-6, S-3), not assumed.
- **"Maximum efficiency regardless of resources."**
  - The biggest efficiency gains are in the harness (async generation and a reward fleet: 1.5–2.3x end to end [V]) and in
    filtering (2–3x fewer rollouts [V]), not in the RL loss.
  - Loss variants mostly change efficiency, not the ceiling [V].

## 5. Decisions for Enzo
**Decided (Oct 24):**
- Several solutions per problem, with a cap. The cap value k is itself an experiment ([06](06-schedule.md) D-2), with
  k = 10 as the default.
- Co-primary endpoints E1 (beam-8) and E2 (avg@16) are adopted now. Their consequences for the design are in
  [05](05-statistics.md) §1.1.
- Licenses and resources are deferred.

**Open:**
| # | Decision | Proposal | File |
|---|---|---|---|
| 1 | Partial-test Codeforces problems | yes, with at least 5 official tests | [01](01-data.md) D2 |
| 2 | Tier D (real inputs, reference oracle) counts as real? | yes for eval robustness; not in the main corpus | [01](01-data.md) D3 |
| 3 | C++ solutions (the route to a time-split eval) | pilot with a length filter | [01](01-data.md) D4 |
| 4 | Framework for confirmatory runs | one-day bake-off, TRL vs one async framework | [02](02-harness.md) §2 |
| 5 | Compiler for RISC-V: clang everywhere, plus a GCC rebuild of GG's RISC-V bench for comparability | approve | [02](02-harness.md) §3.2 |
| 6 | Pre-registration before Phase E | yes, `prereg.md` in this folder | [05](05-statistics.md) §7 |

## 6. First steps (no GPU needed)
1. Instrument the step-time split and the cache hit rate (A-7).
2. Audit MBPP-C's asserts (A-6).
3. Build the stdio and splice modes and the static checks (A-1).
4. Run the Codeforces pilot build on all targets (A-2).
5. Freeze GG-XL (A-3).
6. Run the census and the difficulty predictors on the eval lanes (A-4, A-5).
