# ggrpo: GRPO + (Q)LoRA on Guaranteed Guess (x86 → ARMv8 O2)

Starts from the released `ahmedheakl/gg-armv8-O2`. Each completion gets a binary reward: 1 if its ARM assembly
assembles, links, and passes the task's tests in a macOS 13 VM with GG's toolchain, 0 otherwise. Training is TRL's own
`trl grpo` CLI. The settings and paths are in `config.toml`; TRL's settings are in `grpo.yaml`.

## Layout
| Path | What |
|---|---|
| `config.toml` | all ggrpo settings and paths; any key can be overridden with `--section.key value` |
| `grpo.yaml` | TRL config for `trl grpo --config grpo.yaml` (`TODO` values must be decided before a run) |
| `src/ggrpo/build.py` | builds `data/mbpp_armv8_O2.parquet` in the VM: MBPP-C → x86/arm with GG's commands, validated by the harness |
| `src/ggrpo/data.py` | builds `data/train.jsonl` (MBPP-C O2) and `data/eval.jsonl` (HumanEval-C O2: GG's bench + the authors' tests) |
| `src/ggrpo/harness.py` | assemble → link with the C test driver → run sandboxed → verdict |
| `src/ggrpo/server.py` | FastAPI `POST /score`, runs in the VM |
| `src/ggrpo/reward.py` | the TRL reward function (calls the server) |
| `src/ggrpo/generate.py` | HF `generate` with the paper prompt and decoding, resumable |
| `src/ggrpo/score.py` | scores a completions file through the server; pass@1 on all and on reference-valid tasks |
| `src/ggrpo/paper.py` | runs the released `eval/eval_huamaneval.py` (seeded; optionally `do_sample=False`) in the model-card environment |
| `src/ggrpo/equivalence.py` | log-probs of fixed completions, to compare package environments and attention kernels |
| `docs/toolchain.md` | VM toolchain, data provenance, and the superseded LLM4Decompile HumanEval findings |

## Machines and commands (run everything from this directory)
```bash
# VM (macOS 13.6, Xcode 14.3.1, clang 14.0.3, SDK 13.3; see docs/toolchain.md)
rsync -a --exclude .venv --exclude runs ./ admin@$(tart ip gg-ventura):ggrpo/
uv sync --extra reward
uv run python -m ggrpo.build            # only when the MBPP pool changes; data/mbpp_armv8_O2.parquet is tracked
uv run python -m ggrpo.data
uv run python -m ggrpo.server

# Mac: one reverse tunnel so totoro reaches the VM
ssh -N -R 8765:$(tart ip gg-ventura):8765 totoro.mbzu.ae

# totoro (RTX 6000 Blackwell)
uv sync --extra train
uv run trl grpo --config grpo.yaml                                     # LoRA
uv run trl grpo --config grpo.yaml --load_in_4bit true --bnb_4bit_quant_type nf4 \
    --run_name qlora --output_dir runs/qlora                           # QLoRA
uv run python -m ggrpo.generate --generate.adapter runs/lora --generate.out runs/lora/beam_search.jsonl
uv run python -m ggrpo.score --score.predictions runs/lora/beam_search.jsonl --score.out runs/lora/beam_search.scored.jsonl
```
- **Reference preflight** (defines the reference-valid set; all 164 with the authors' tests):
  `uv run python -m ggrpo.score --score.predictions data/eval.jsonl --score.column arm_ref --score.out runs/preflight/eval_ref.jsonl`
- **Logs:** JSON lines on stderr and in `runs/ggrpo.jsonl`. Read them with `hl -F runs/ggrpo.jsonl` in a tmux pane,
  or pipe a command through `| hl -P`.
- **Training curves (Trackio, local only):**
  `ssh -L 7860:127.0.0.1:7860 totoro.mbzu.ae 'cd ggrpo && TRACKIO_DIR=runs/trackio uv run trackio show --project cs8001'`
- **Tests:** `uv run pytest`. `tests/test_harness.py` is authoritative only inside the VM.

## Verified (Oct 22, 2026)
- **Toolchain:** official HumanEval-C O0/O2 recompiles byte-identically for 316/328 rows, and BringUp for 131/132.
  These were not data errors. Our C came from LLM4Decompile, whose problem order differs for 6 problems
  (`docs/toolchain.md`). The authors' own sources (HF `asmbench_source`) recompile to all 328 rows.
- **Rewritten harness, in the VM:**
  - HumanEval references pass 164/164 with the authors' tests. With LLM4Decompile's drivers they pass 158/164, failing on the
    same 6 problems as the old harness.
  - All 952 MBPP references pass.
- **MBPP build:** `ggrpo.build` reproduces the first MBPP-C O2 build byte for byte (x86, arm, driver, verdict, 973 rows),
  and `data/train.jsonl` is unchanged.
  - On the 164 beam-8, 16k-cap predictions, the old and new harnesses give the same verdict for every task.
- **Prompt fidelity:** all 164 prompts are byte-identical to the paper's f-string, and their token counts equal the beam-8, 16k-cap run.
- **Extraction:** the regex in `config.toml` equals the paper's `split` on edge cases and on all 164 beam-8, 16k-cap outputs.
- **TRL:** `grpo.yaml` parses with TRL 1.16. `TODO` values are rejected, and `reward_pass` works against the live server.
- **Baseline** (`data/raw/humaneval_beam8_cap16k.jsonl`: beam 8, `repetition_penalty` 1.1 inherited, 16000-token cap, VM): **72/164 = 43.9%** with the
  authors' tests. With LLM4Decompile's drivers it is 68/164 (68/158). The paper reports 45.12% = 74/164.
  This run hit the 16000-token cap on some tasks; the paper's cap is `32000 - prompt tokens`.
