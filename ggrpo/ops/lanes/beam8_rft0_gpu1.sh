#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER): the validation beam-8 eval of online RFT seed 0 at step 368, both halves one after the
# other on the same card, the same lane as beam8_seed1_gpu1.sh.
#
# It waits for that lane's LANE_EXIT first: both lanes take gpu1.lock per half, so starting now could slip this lane in
# between seed 1's two halves.
#
# The model is runs/rft_seed0_368: hard links to runs/rft_seed0's final save_model output (PASS2_EXIT 0,
# 2026-11-20T16:44:45Z). Generation only; scoring is a separate step.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
mkdir -p runs/codenet_val_rft0_368
S=runs/codenet_val_rft0_368/lane.status
echo "WAIT seed1 lane $(date -u +%FT%TZ)" >> $S
until grep -q "^LANE_EXIT" runs/codenet_val_seed1_368/lane.status; do sleep 120; done
echo "LANE_START $(date -u +%FT%TZ)" >> $S
for half in gpu1 ponyo; do
  echo "HALF_START $half $(date -u +%FT%TZ)" >> $S
  flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh $half "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" \
    runs/rft_seed0_368 runs/codenet_val_rft0_368
  echo "HALF_EXIT $? $half $(date -u +%FT%TZ)" >> $S
done
echo "LANE_EXIT $(date -u +%FT%TZ)" >> $S
