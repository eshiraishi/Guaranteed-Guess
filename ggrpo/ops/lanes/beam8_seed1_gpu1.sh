#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER): the validation beam-8 eval of GRPO seed 1 at step 368 (prereg H1 endpoint E1), both
# halves one after the other on the same card.
#
# Seed 0's two halves ran on gpu1 and on ponyo, which are the same card model, so running both of seed 1's halves here
# keeps the one-GPU-type rule and needs no second host. That matters today: ponyo left the tailnet at
# 2026-11-19T19:39Z and is still unreachable, and its disk holds seed 0's ponyo-half predictions. totoro is reached
# directly at totoro.mbzu.ae from the campus network, which is how this runs at all.
#
# The model is runs/grpo_seed1_368: hard links to runs/codenet_seed1's final save_model output (PASS2_EXIT 0,
# 2026-11-19T18:09:06Z), the same export shape as runs/codenet_grpo368 for seed 0.
#
# Generation only. Scoring goes through the single reward VM that the online-RFT run is driving, so it is a separate
# step: scoring here would slow every training step for as long as it lasted.
# The two halves take the same gpu1.lock, so the second waits for the first instead of sharing the card.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
S=runs/codenet_val_seed1_368/lane.status
echo "LANE_START $(date -u +%FT%TZ)" >> $S
for half in gpu1 ponyo; do
  echo "HALF_START $half $(date -u +%FT%TZ)" >> $S
  flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh $half "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" \
    runs/grpo_seed1_368 runs/codenet_val_seed1_368
  echo "HALF_EXIT $? $half $(date -u +%FT%TZ)" >> $S
done
echo "LANE_EXIT $(date -u +%FT%TZ)" >> $S
