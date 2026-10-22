# CS8001: GRPO/DAPO with QLoRA on Guaranteed Guess. Status and plan (Oct 22)

## Goal
Start from the released GG model `ahmedheakl/gg-armv8-O2` (Qwen2.5-Coder-1.5B, already SFT'd on x86 to ARMv8 O2) and improve it with RL using test-based rewards (GRPO/DAPO via TRL). Compare LoRA against QLoRA. Evaluate on GG's official HumanEval-C O2. W&B project: `cs8001`.

## Where we are (done)
| Item | Result |
|---|---|
| Toolchain | Tart VM `gg-ventura`: macOS 13.6, Xcode 14.3.1, Apple clang 14.0.3, SDK 13.3. Reproduces GG's released assembly byte-for-byte: HumanEval 316/328 rows, BringUp 131/132 rows. |
| Why mismatches | HumanEval problems 0, 1, 101, 125, 137, 162 are errors in GG's release (problem 0 has `return 10;`; the other five hold the next problem's function). BringUp `satomi` O2 unexplained (probably source). Not a toolchain issue. |
| Harness in VM | Official reference ARM passes 158/164; empty prediction fails 164/164. |
| RL prompts | MBPP-C O2 built in the VM: 952 problems pass validation (reference passes, empty fails). New file with toolchain column, old macOS 27 file kept as `*_clang21`. |
| Paper read | No compiler version stated; ARMv8 compiled on a physical M2 Pro. HumanEval O2 45.12%, BringUp O2 15.38% (beam 8). |

All new files are in `rl_simple/` on the Mac. Nothing pushed to GitHub yet.

## Blockers / waiting
1. **Step 0 generation** is running on `ponyo` (RTX 4070 Ti SUPER, tmux session `cs8001-step0`, log `~/cs8001-step0/full.log`): unmodified model, 164 official prompts, beam 8, 16k-token cap. Scoring in the VM follows. `totoro` stays off limits until Enzo says otherwise.
2. **Authors** (not blocking): Xcode/clang version confirmation, the broken-rows observation (problems 1, 101, 125, 137, 162 hold the next problem's function; problem 0 has `return 10;`), and whether MBPP/ExeBench/HumanEval/BringUp were in the SFT data.
3. **Decisions by Enzo:** headline denominator (all 164 vs the 158 valid), GRPO vs DAPO loss (discuss before any training), and the commit layout.

## Plan, in order
1. Step 0: generate on `ponyo` (3-problem benchmark first), score in the VM, report pass@1 on 164 and 158 (beam 8, to compare with the paper's 45.12%).
2. Overlap check of MBPP-C against GG's public SFT data (assembly text match). ExeBench later, only if needed.
3. Write the TRL GRPO script (LoRA bf16 first, QLoRA second), `rl_simple/train.py`, reward function calling the VM reward server. Smoke test (10 steps) before any long run.
4. Cap completion length and exclude truncated outputs (model sometimes loops). Decide GRPO vs DAPO loss first.
5. Training on the Blackwell GPU (hostname/access pending), one tunnel to the Mac VM reward server.
6. Evaluate LoRA and QLoRA on the official 158/164 set with the same sampled pass@1 used in training; beam 8 as a later check.

## Not doing now
ExeBench, O3, BringUp as training data, the 20-arm pre-registered design, FSDP, custom trainer/sync code, the paper draft.
