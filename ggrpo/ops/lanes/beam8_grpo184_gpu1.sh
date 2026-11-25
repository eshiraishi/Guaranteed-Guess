#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER), under runs/gpu1.lock: the 192-task half of the GRPO step-184 checkpoint's validation
# beam-8 eval (Enzo, Nov 18: "Yes, generate now").
#
# GPU1, not the Blackwell: GG's 65/383 and the full-data SFT's 95/383 were both measured on the 4070s, and a comparison
# keeps one GPU type. GPU0 is busy with the run's second pass and is never shared.
#
# Generation only. Scoring goes through the single reward VM that pass 2 is driving, so it is held for a separate step
# rather than run here, where it would slow every training step for as long as it lasted.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src
flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh gpu1 "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" \
  runs/codenet_seed0/checkpoint-184 runs/codenet_val_grpo184
