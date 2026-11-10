# 02. Harness: what runs today

**Status (Nov 10, 2026): in use.** Only the stack we run now. Tags as in [00](00-overview.md).

## 1. Training
- **Trainer:** TRL 1.16 (`trl sft`, `trl grpo`) on totoro GPU0, venv `~/ggrpo/.venv` (never modified in place) [R].
- **Generation:** vLLM 0.31 colocated with the trainer on the same GPU (synchronous, exactly on-policy) [R].
- **Precision for RL:** fp16 in both vLLM and the trainer, fp32 master weights with loss scaling (`python -m ggrpo.fp16`
  writes the fp16 model folder). In bf16 the median sequence importance ratio fell to 0.03 above 8k tokens; in fp16 it
  stays near 1 (Nov 10, 64 RL prompts) [R].
- **Importance-sampling correction:** TRL default (sequence mask, cap 3) [R].
- **Tracking:** W&B `assembly-transpilation/rlvr`, readable run ids, no plan IDs [R].

## 2. Rewards
- One macOS 13.6 VM (`gg-ventura`, Xcode 14.3.1, 10 CPUs, 8 workers) on the Mac serves `POST /score` [R].
- totoro reaches it through an SSH tunnel (totoro:8765); `tunnel_guard.sh` frees the port after a Mac DNS drop, which
  otherwise stalls training silently [R].
- The Mac must stay awake and online for every RL run and every scoring job.
- Worst scoring batch so far: ~86 s [R]. The server returns 503 below 5 GB free disk and deletes ld snapshots [R].

## 3. Verification modes
| Mode | Unit | Oracle | Used by |
|---|---|---|---|
| driver | one function (`func0`) | C driver with asserts | MBPP-C, HumanEval-C |
| stdio | whole program | statement samples (RL, E1/E2) or AOJ official tests (E8), exact compare up to trailing whitespace | CodeNet |
| BringUp | whole program | upstream `make run-tests` | BringUp |

Rules in force: binary reward (pass all tests); stdio tasks need ≥ 2 distinct expected outputs; sandbox-exec with CPU,
memory and file-size limits; the toolchain reproduces GG's references byte for byte on 316/328 HumanEval-C and 131/132
BringUp rows, and the authors' tests pass 164/164 references [R].

## 4. Evaluation lanes
| Lane | Hardware | Stack | Jobs |
|---|---|---|---|
| totoro GPU1, ponyo | 4070 Ti SUPER 16 GB | torch 2.13, transformers 5.17, vLLM | checkpoint monitoring, census |
| student lab (Slurm) | RTX 5000 Ada 32 GB, ≤ 2 jobs × 2 GPUs | transformers 4.50 view dirs (rope_theta assert) | BringUp; final scoring if the benchmark shows it faster |

**One GPU type per comparison:** every arm, and the SFT start, are scored on the same GPU type and stack; a lane moves
whole or not at all [R rule].

## 5. Reproducibility
- Record config, data hash, seed and git SHA for every run (W&B logs the config) [R].
- Assert `rope_theta` at load time in every generator (transformers 4.x silently uses 10000 on 5.x checkpoints) [R].
- Release per-task outputs and scores for every reported number.
