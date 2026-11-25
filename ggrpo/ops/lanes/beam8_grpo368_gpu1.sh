#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER), under runs/gpu1.lock: the 192-task half of the finished GRPO run's validation beam-8
# eval -- step 368, the headline number (prereg H1 endpoint E1). Chained behind this card's step-184 half.
#
# The model is runs/codenet_grpo368: hard links to the final save_model output of runs/codenet_seed0 (PASS2_EXIT 0,
# 2026-11-18T15:46:31Z), so the folder holds only the files a loader needs and ponyo can rsync it whole. Same bytes on
# both hosts, sha256 0d05f384de69cb80ccaa15f845221e9eeacbf7d023ad9fc4b796592a29662762.
#
# GPU1, not the Blackwell: GG's 65/383 and the full-data SFT's 95/383 were both measured on the 4070s, and a comparison
# keeps one GPU type. The Blackwell is running the permuted-reward control.
#
# Generation only. Scoring goes through the single reward VM the control run is driving, so it is held for a separate
# step rather than run here, where it would slow every training step for as long as it lasted.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
S=runs/codenet_val_grpo368/gpu1.status
echo "WAIT step-184 gpu1 half $(date -u +%FT%TZ)" >> $S
until grep -qE "^(LANE_DONE|FAILED|SHORT)" runs/codenet_val_grpo184/gpu1.status; do sleep 120; done
grep -q "^LANE_DONE" runs/codenet_val_grpo184/gpu1.status \
  || { echo "STEP184_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh gpu1 "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" \
  runs/codenet_grpo368 runs/codenet_val_grpo368
