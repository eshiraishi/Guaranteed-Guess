#!/usr/bin/env bash
# ponyo GPU0 (4070 Ti SUPER): the 191-task half of the GRPO step-184 checkpoint's validation beam-8 eval, the same half
# this card ran for the GG base and for the full-data SFT.
# Overlay venv ~/gg/ggrpo-pilot/.venv-beam (transformers 5.17, as on totoro): the paper venv here is 4.50, which
# ignores rope_parameters in a checkpoint saved by transformers 5.
# Started by ops/lanes/ship_grpo184.sh once ponyo has pulled the weights from totoro.
cd ~/gg/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/gg/ggrpo/src
runs/codenet_val_beam8_lane.sh ponyo "$HOME/gg/ggrpo" "$HOME/gg/ggrpo-pilot/.venv-beam/bin/python" \
  runs/codenet_seed0/checkpoint-184 runs/codenet_val_grpo184
