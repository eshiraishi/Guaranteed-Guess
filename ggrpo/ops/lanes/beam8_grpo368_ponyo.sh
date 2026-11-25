#!/usr/bin/env bash
# ponyo GPU0 (4070 Ti SUPER): the 191-task half of the finished GRPO run's validation beam-8 eval -- step 368, the
# headline number. The same half this card ran for the GG base, the full-data SFT and step 184.
# Chained behind this card's step-184 half; the weights were already pulled from totoro (hash checked) while that half
# was still generating, so nothing here depends on the Mac being awake.
# Overlay venv ~/gg/ggrpo-pilot/.venv-beam (transformers 5.17, as on totoro): the paper venv here is 4.50, which
# ignores rope_parameters in a checkpoint saved by transformers 5.
cd ~/gg/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/gg/ggrpo/src TQDM_DISABLE=1
S=runs/codenet_val_grpo368/ponyo.status
echo "WAIT step-184 ponyo half $(date -u +%FT%TZ)" >> $S
until grep -qE "^(LANE_DONE|FAILED|SHORT)" runs/codenet_val_grpo184/ponyo.status; do sleep 120; done
grep -q "^LANE_DONE" runs/codenet_val_grpo184/ponyo.status \
  || { echo "STEP184_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
runs/codenet_val_beam8_lane.sh ponyo "$HOME/gg/ggrpo" "$HOME/gg/ggrpo-pilot/.venv-beam/bin/python" \
  runs/codenet_grpo368 runs/codenet_val_grpo368
