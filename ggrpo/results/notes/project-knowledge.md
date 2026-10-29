# CS8001 / Guaranteed Guess: everything we know so far (temporary working file, not committed)

Written Oct 22, 2026 from the work session on Enzo's Mac. Facts are marked **verified** (we measured them), **paper** (stated in the
GG paper), or **inferred** / **open**. See also `status-and-plan.md` and `plan-history.md` (committed on `feature/env`).

## 1. Goal and context
- Project: improve the released GG model `ahmedheakl/gg-armv8-O2` (Qwen2.5-Coder-1.5B-Instruct, SFT'd x86 -> ARMv8 O2 assembly) with RL
  using test-based rewards (GRPO/DAPO via TRL), comparing LoRA vs QLoRA. Evaluation target: GG's official HumanEval-C O2. W&B project `cs8001`.
- Original paper: "Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees"
  (Heakl et al., EMNLP Findings 2025, arXiv 2506.14606).
- Because the numbers are delicate, the decision (Enzo) is that everything turning C into x86/ARM assembly must use the same toolchain as GG's
  released data, and whatever we build now must stay reusable so numbers don't change later.

## 2. Repos, branches, machines
- Mac (macOS 27.0.1, Apple M5 Pro, 24 GB): `~/git/Guaranteed-Guess`, branch `qlora-grpo-study` (c5b4c60, two study commits on top of
  `main`, NOT on GitHub). Uncommitted there: modified `rl/data/build_exebench.py`, `rl/reward/harness.py`, `rl/reward/server.py`; new
  `rl/analysis/compiler_drift.py`, `rl/data/build_bringup.py`, `rl/eval/e2e_mac.py`; `rl_simple/` (all my work files). `rl/artifacts/` and
  `data_cache/` are gitignored (about 109 MB and 810 MB).
- Worktree `~/git/Guaranteed-Guess-feature-env`: branch `feature/env`, ONE commit `eca8c69` on top of latest `origin/main` (89cf7a4), PUSHED to
  github.com/eshiraishi/Guaranteed-Guess. Contents: 18 files under `rl_simple/` (scripts, toolchain record, mismatch notes, `tasks_O2.jsonl`,
  `step0_ref_empty_O2.jsonl`, MBPP-C O2 parquet, harness copy, README, the two plan docs). No trailers in the commit message. Not in that commit:
  the step-0 results (`step0_result.md`, `step0_scored_O2.jsonl`, `step0_preds_raw_ponyo_O2.jsonl`), the later `gen.py --rep` option, the big
  MBPP jsonl files, the old clang-21 MBPP parquet.
- ponyo (`ssh ponyo`, resolves to ponyo.mbzu.ae, user `eshiraishi`, key auth): RTX 4070 Ti SUPER 16 GB, 60 GB RAM, Python 3.12, tmux 3.4, no
  python3-venv or pip system-wide. Another user (`jbakita`) is logged in; another user (`ruslahnq`) runs an NVIDIA MPS daemon. Work dir
  `~/cs8001-step0/`. tmux session `cs8001-step0`.
- totoro.mbzu.ae: Enzo said SSH is available (Oct 22 06:30); we have NOT connected to it and have not been asked to.
- Reward/VM host: the Mac itself, Tart VM `gg-ventura`.

## 3. The pinned toolchain (verified)
- Tart 2.41.1 installed from the release tarball at `~/tools/tart/tart.app/Contents/MacOS/tart` (Homebrew formula was broken). PATH not changed.
- Image `ghcr.io/cirruslabs/macos-ventura-xcode:14.3.1`, digest `sha256:ca83d14f4399a6587e828a18bfe2392f9f61d62692d03781fdd3e0083cafc502`
  (100 GB disk, 63 GB used). Local VM clone: `gg-ventura`. Run headless: `tart run --no-graphics gg-ventura` (it had stopped once and was restarted).
- Inside: macOS 13.6 (22G120), arm64, Xcode 14.3.1 (14E300c), Apple clang 14.0.3 (clang-1403.0.22.14.1), SDK 13.3. System python3 3.9.6.
  Output carries `.build_version macos, 13, 0  sdk_version 13, 3`, same as GG's released data.
- Access: key-based SSH as `admin@<tart ip>` (usually 192.168.64.2). `tart exec` (guest agent) does not work in this image. Enzo installed the
  public key himself; I never typed a password (rule: I don't enter passwords; the image has a documented default login).
- The Mac's own toolchain is NOT the reference: Apple clang 21.0.0, SDK 27.0, CLT only (no Xcode). All data built locally before Oct 22 carries
  `.build_version macos, 27, 0` and differs from GG's (0% exact x86 match, about 55% similarity on O2 prompts).
- Compile commands (from `train/compile_{x86,arm}_clang.py` and `rl/data/build_humaneval.py`): `clang -arch arm64 -S -O{0,2,3}` and
  `clang -S -target x86_64-apple-darwin -O{0,2,3}`. BringUp adds `-DTARGET_HOST -I common -I target -I <bench> -Wno-strict-aliasing`.

## 4. Verification results (all verified in the VM)
- HumanEval-C official ARMv8 O0/O2 (`ahmedheakl/gg-bench-armv8-O{0,2}`, 328 rows): recompiling the LLM4Decompile C source gives
  byte-identical x86 and arm for 316/328. The 12 mismatches are 6 problems at O0 and O2:
  - problem 0: official C has `return 10;` where ours has `return 0;` (patching the C reproduces the official asm exactly; `p0_official_variant.c`).
  - problems 1, 101, 125, 137, 162: the official rows are exact copies of the NEXT problem (2, 102, 126, 138, 163); the release has duplicate pairs and
    the real function for problem k is missing. Their official reference ARM therefore crashes the tests.
  - Default flags reproduce everything else, so no flag sweep was needed.
- BringUpBench official O0/O2 (`ahmedheakl/gg-bench-bringup-O{0,2}`, 66 programs, 132 rows): 131/132 byte-identical; only `satomi` O2 differs
  (official is much larger; probably a source difference; not investigated).
- Harness in VM, HumanEval O2 (164 official rows): reference ARM passes 158/164 (the 6 above fail with crash); an empty prediction fails 164/164.
- Inference about the authors' machine (open, not confirmed): SDK 13.3 means Xcode/CLT 14.3.x (14.3 and 14.3.1 use the same clang 14.0.3); an M2 Pro
  runs macOS >= 13.2. clang output is hardware independent, so a bare-metal Mac is not needed.

## 5. The datasets
- Released GG datasets (`ahmedheakl/gg-bench-*`) contain only `x86`, `arm`, `file` columns: assembly only, no C source, no tests. Tests come from
  LLM4Decompile `decompile-eval` (HumanEval-C and MBPP-C asserts) and BringUp Makefiles. HumanEval-C = 164 problems; the gg-bench `file` path is
  `eval/problem{k+1}/code.c`.
- GG's SFT set: ~1.31M pairs from AnghaBench (1.01M) + The Stack v2 (306k), O0 and O2 only, no tests (so unusable as an RL reward source).
- `rl/artifacts/data/` (gitignored, macOS-27 builds, STALE for toolchain): humaneval 492 rows (328 official + 164 regenerated O3), mbpp 2919 (973 x
  O0/O2/O3), exebench train 30,509 / valid 2,828 / test 2,810.
- MBPP-C O2 rebuilt in the VM (verified): 974 problems, 1 fails to compile, 973 kept; 952 reference-pass, 21 fail (20 crashes + mbpp/579 link
  error); empty fails on all. Versus the old macOS-27 build (973 rows, 951 pass): a strict superset, plus mbpp/858 now passes. Failing set: mbpp/11,
  253, 304, 318, 327, 329, 336, 340, 380, 395, 402, 439, 579, 583, 585, 642, 718, 773, 793, 878, 929. File: `rl_simple/mbpp_armv8_O2_ventura.parquet`
  (has a `toolchain` column).
- Not yet rebuilt in the VM: HumanEval O3, ExeBench train/valid/test, BringUp O3.
- Contamination status (open): MBPP-C, ExeBench not stated in GG's training data; GG sets are GitHub-derived, so overlap with ExeBench is possible
  but unmeasured. HumanEval O3 almost certainly unseen (GG trained and reported O0/O2 only). Plan: assembly-text overlap check of MBPP-C against GG's
  public SFT data; ExeBench later if needed. Overlap with an eval set = contamination; overlap with only the RL pool = less learning signal.

## 6. What the paper says (read in full, camera-ready)
- ARMv8 compiled natively on an Apple M2 Pro (macOS) with clang; no Xcode/clang/SDK versions, no VM, no extra flags; x86 machine not stated.
- Table 3 (GG-1.5B, ARMv8): HumanEval O0 99.39%, O2 45.12% (= 74/164); BringUpBench O0 49.23%, O2 15.38%. Baselines (Qwen, StarCoder2,
  DeepSeek-R1) ~0%; GPT-4o 10.3% / 4.24% on HumanEval ARMv8 O0/O2. GG-0.5B: 86.06/25.45 (HumanEval), 27.69/3.08 (BringUp).
- Decoding/eval: pass@1, beam search with beam size 8, no sampling; greedy about 98.18% vs beam 8 99.39% at O0 (Table 5). Correct only if all tests
  pass. 16k training context, RoPE extrapolation to 32.7k at inference, "extended tokenizer" (opcode/register tokens). vLLM for deployment.
  Training: A100 40 GB, lr 2e-5, cosine, 2 epochs, 3 days (text says effective batch size 2, probably a typo).
- BringUp: text says 65 programs (dataset has 66 per level); mean ~10.5k tokens x86; 17% of BringUp failures are context truncation.
- Inconsistencies: section 5.3 says "trained only on -O2", section 3.1 says O0 and O2. Nothing on MBPP/ExeBench or which functions were in SFT data.

## 7. Step 0: baseline on official HumanEval-C O2 (verified)
- Setup: unmodified `ahmedheakl/gg-armv8-O2`, bf16, transformers 5.20.2 / torch 2.14.1 (cu130), chat template, prompt
  `Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{asm}\n```` , beam 8, no sampling, max_new_tokens 16000,
  prediction = text between ```armasm and ```. Generated on ponyo (1.88 h), scored in the VM with the unmodified harness.
- Result (run 1): pass@1 **68/164 = 41.46%** (paper 45.12%); on the 158 valid problems **68/158 = 43.04%**. Verdicts (164): pass 68, crash 79,
  assemble_err 16, timeout 1; 7 generations hit the 16k cap and all fail to assemble. The 6 invalid problems all fail.
- Investigating the 3.7-point gap: the checkpoint's `generation_config.json` has `repetition_penalty 1.1` (plus temperature 0.7, top_k 20,
  top_p 0.8, do_sample true); HF `generate` still applies the penalty with do_sample=False. Run 2 (`--rep 1.0`, same everything else, output
  `~/cs8001-step0/preds_rep1_raw.jsonl`, log `rep1.log`) FINISHED: 155/164 generations identical to run 1, same verdict counts, **68/164 = 41.46%
  again (no problem gained or lost)**. So the repetition penalty is not the cause. RoPE extrapolation is unlikely to matter (nearly all prompt+output < 16k),
  and the released tokenizer is the stock Qwen2.5 one (vocab 151,936), so the paper's extended tokenizer can't be tested. Remaining candidates:
  unstated differences in the authors' evaluation, the 7 runaway beams, beam-search implementation/library differences, bf16 noise.
- Mac MPS benchmark (stopped on Enzo's request): beam 8 on three prompts took 117 s, 135 s, 48 s; unreliable and about 15x slower than ponyo.

## 8. Reward harness (`rl/reward/harness.py`, copied unmodified into `rl_simple/reward_copy/`)
- Assembles the predicted `.s` with `clang -arch arm64 -c -x assembler`, links with a pre-compiled test driver, runs it (sandbox-exec profile,
  CPU/wall limits) and returns a verdict: pass, empty, assemble_err, link_err, crash, timeout, wrong_output, infra_err. The C source of the function
  is never compiled into the test binary, so empty/truncated outputs fail (hack-proof). Note: the copy in use includes uncommitted edits (7 added,
  2 removed lines) on the study branch.
- Scripts in `rl_simple/`: `export_tasks.py` (official parquet rows -> JSONL), `vm_check.py` (run in VM: reference+empty check, or score
  predictions), `vm_build.py` + `export_mbpp_jobs.py` + `make_mbpp_parquet.py` (MBPP-C build), `gen.py` (beam-8 generation, `--rep` option in the
  working copy only).

## 9. Gotchas learned
- ponyo: another user's NVIDIA MPS daemon makes `torch.cuda` fail with error 805. Workaround local to our process:
  `export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off`. No sudo, no daemon changes.
- ponyo has no python3-venv/pip: venv created with `--without-pip`, pip bootstrapped with get-pip.py inside `~/cs8001-step0/venv`.
- The VM's remote shell is zsh: unquoted `$F` isn't word-split; run loops via `bash -c`.
- tmux sessions belong to a user; the session `cs8001-step0` is under `eshiraishi`.
- Don't Ctrl-c in the tmux session (stops generation); detach with Ctrl-b d.
- Tool rules in this session: no entering passwords, no SSH to hosts until Enzo says so directly (totoro/ponyo were each approved separately).

## 10. Open questions and decisions
- Headline denominator: all 164 vs 158 valid (Enzo to decide). Handling of the 6 invalid problems: exclude, or add our own VM-compiled versions
  labelled regenerated.
- GRPO vs DAPO loss (completion length matters; Enzo wants to discuss before any training). Runaway generations: cap completion length and exclude
  truncated outputs.
- Questions for the authors (nothing blocks us): Xcode/clang version and physical vs VM; exact ARMv8/x86 compile flags; the `code.c` for problems
  1, 101, 125, 137, 162 and the `return 10` in problem 0 (and BringUp `satomi` O2); optional: O3/MBPP/ExeBench data, SFT overlap. Message text was
  drafted and given to Enzo.
- Report/paper updates (limitations.tex, setup.tex, datacard.tex todo) are skipped for now; the old paper text still says "Apple clang 21".

## 11. Plan (see status-and-plan.md)
1. Finish step-0 variant and decide whether 41.5% (43.0% on 158) is the accepted baseline.
2. MBPP overlap check vs GG's SFT data. 3. Write TRL GRPO script (LoRA bf16, then QLoRA NF4) in `rl_simple/train.py`, reward function calling the
   VM reward server; 10-step smoke test first. 4. Train on the Blackwell GPU (access pending), one tunnel to the Mac VM. 5. Evaluate LoRA/QLoRA on the
   official 158/164 with the same sampled pass@1, beam 8 later as a check. Not doing now: ExeBench, O3, BringUp as training data, the 20-arm
   pre-registered design, FSDP, custom trainer/sync code, the paper draft.
