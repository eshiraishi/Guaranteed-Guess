# Toolchain and data provenance

## VM (the reward harness and every data build run here)
- gg-ventura: tart clone of `ghcr.io/cirruslabs/macos-ventura-xcode:14.3.1`,
  image digest `sha256:ca83d14f4399a6587e828a18bfe2392f9f61d62692d03781fdd3e0083cafc502`
- macOS 13.6 (22G120), arm64
- Apple clang 14.0.3 (clang-1403.0.22.14.1), Xcode 14.3.1 (14E300c), SDK 13.3

With these, GG's official HumanEval-C O0/O2 rows recompile byte-identically from the authors' sources
(HF `ahmedheakl/asmbench_source` @ 89bca970) for all 328 rows, and BringUp for 131/132.

## MBPP-C O2 (training pool, `python -m ggrpo.build`)
- C source and asserts: HF `LLM4Binary/decompile-eval` @ b9271fae, `mbpp` split, `language == "c"`, `opt == "O0"`
  rows (974; the C is the same at every opt, only its asm differs). Stored as `data/raw/decompile_eval_mbpp.parquet`.
- Compile commands are GG's (`train/compile_x86_clang.py`, `train/compile_arm_clang.py`) at `-O2`. One task
  (`mbpp/217`) does not compile and is dropped. Of the other 973, the reference arm passes its driver on 952. All 973
  drivers fail to link a `.s` without the function, so no driver defines `func0` itself.
- The rebuild with `ggrpo.build` is byte-identical (x86, arm, driver, verdict) to the first build made with the
  pre-package scripts (commit eca8c69, `rl_simple/vm_build.py`).

## HumanEval-C with LLM4Decompile's C (superseded)
Our first HumanEval tasks took the C and asserts from LLM4Decompile, not from the authors. Those recompile to GG's rows for
158/164 problems. The 6 differences are not errors in GG's release. LLM4Decompile's problem order differs from GG's for
these problems:
- `humaneval/0`: GG's asm equals our compile of the same C with the final `return 0;` changed to `return 10;`
  (byte-identical for x86 and arm, at O0 and O2). GG's reference therefore fails the LLM4Decompile assert, which
  expects 0 when no two elements are close.
- `humaneval/1, 101, 125, 137, 162`: GG's rows equal our compile of the next problem (2, 102, 126, 138, 163).

With those drivers the references passed 158/164, and the beam-8, 16k-cap run scored 68/164. With the authors' tests
the references pass 164/164 and that run scores 72/164. ggrpo evaluates only with the authors' tests.
