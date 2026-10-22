# rl_simple: step-0 baseline tooling in the GG toolchain VM

Everything here assembles, links and runs inside a macOS 13 Tart VM so the assembly matches the data released
with Guaranteed Guess (EMNLP Findings 2025, arXiv 2506.14606), which carries `.build_version macos, 13, 0  sdk_version 13, 3`.

## Toolchain (recorded in `step0_toolchain.txt`)
- Tart image `ghcr.io/cirruslabs/macos-ventura-xcode:14.3.1`, digest
  `sha256:ca83d14f4399a6587e828a18bfe2392f9f61d62692d03781fdd3e0083cafc502`
- macOS 13.6 (22G120), arm64; Xcode 14.3.1 (14E300c); Apple clang 14.0.3 (clang-1403.0.22.14.1); SDK 13.3
- Tart 2.41.1 installed from the release tarball (the Homebrew formula was broken); the VM python3 (3.9.6) is enough.

## Run the VM check
```
tart clone ghcr.io/cirruslabs/macos-ventura-xcode:14.3.1 gg-ventura && tart run --no-graphics gg-ventura
# copy rl_simple/ to the VM (ssh key auth), then inside the VM:
python3 vm_check.py tasks_O2.jsonl ref_empty_O2.jsonl           # official reference ARM + empty prediction
python3 vm_check.py tasks_O2.jsonl preds.jsonl scored.jsonl     # score predictions ({"task_id","pred"} per line)
python3 vm_build.py mbpp_jobs_O2.jsonl out.jsonl 4              # compile + validate MBPP-C (jobs via export_mbpp_jobs.py)
```
`reward_copy/` is a copy of `rl/reward/harness.py`, `sandbox.sb` and `include/` from the `qlora-grpo-study` branch working tree,
used as-is for all results below (harness.py there carries uncommitted edits of 7 added / 2 removed lines relative to that branch's HEAD).
`gen.py` generates with `ahmedheakl/gg-armv8-O2` (chat template, paper prompt, beam search, no sampling); `export_tasks.py` reads the
HumanEval parquet from the study branch.

## Verified results
- Official HumanEval-C ARMv8 (O0/O2, 328 rows) recompiled in the VM: 316/328 byte-identical for both x86 and arm.
- Official BringUpBench ARMv8 (O0/O2, 132 rows): 131/132 byte-identical (`satomi` at O2 differs).
- Reference ARM, HumanEval-C O2: 158/164 pass in the VM; the empty prediction fails 164/164 (`step0_ref_empty_O2.jsonl`).
- The 6 HumanEval mismatches are data issues, not toolchain: problem 0 has `return 10` instead of `return 0`
  (`p0_official_variant.c` reproduces the official asm byte for byte); problems 1, 101, 125, 137, 162 are copies of the next
  problem (`mismatch_findings.md`).
- MBPP-C O2 rebuilt in the VM (`mbpp_armv8_O2_ventura.parquet`, with a `toolchain` column): 973 rows, 952 reference-pass, empty fails on all.
