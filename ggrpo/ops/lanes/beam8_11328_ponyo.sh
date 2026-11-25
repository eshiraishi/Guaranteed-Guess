#!/usr/bin/env bash
# ponyo GPU0 (4070 Ti SUPER): the start check's 191-task half, the same half this card ran for the GG base and for the
# 30% checkpoint. Overlay venv ~/gg/ggrpo-pilot/.venv-beam (transformers 5.17, as on totoro): the paper venv here is
# 4.50, which ignores rope_parameters in a checkpoint saved by transformers 5. Started by ops/lanes/ship_11328.sh once
# the Mac has streamed the model across; generation only, scored from the Mac afterwards.
cd ~/gg/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/gg/ggrpo/src
runs/codenet_val_11328/lane.sh ponyo "$HOME/gg/ggrpo" "$HOME/gg/ggrpo-pilot/.venv-beam/bin/python"
