#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER): the ponyo half (191 tasks) of GRPO seed 0 step 368's validation beam-8 eval (E1).
# The gpu1 half ran on gpu1 on Nov 20; the ponyo half's predictions sit on ponyo, which has been offline since
# 2026-11-19T19:39Z. ponyo is the same card model, so re-running that half here keeps the one-GPU-type rule.
# Runs before the RFT beam-8 lane (GRPO first: this is the missing piece of H1's E1). Generation only.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
O=runs/codenet_val_grpo368_ponyohalf; mkdir -p $O; S=$O/lane.status
cp runs/codenet_val_grpo368/pristine_ponyo.jsonl $O/pristine_ponyo.jsonl
cp runs/codenet_val_grpo368/pristine_ponyo.jsonl $O/val_ponyo.jsonl
echo "LANE_START $(date -u +%FT%TZ)" >> $S
flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh ponyo "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" \
  runs/codenet_grpo368 $O
echo "LANE_EXIT $? $(date -u +%FT%TZ)" >> $S
