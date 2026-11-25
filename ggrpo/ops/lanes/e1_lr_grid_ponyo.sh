#!/usr/bin/env bash
# ponyo GPU0 (4070 Ti SUPER): prereg E1 (beam 8) for the imitation arm's three LRs, this card's 191-task half -- the
# same half it ran for the GG base, the full-data SFT and both GRPO checkpoints.
#
# Why this card gets E1 and not part of E2: ggrpo.sample needs vLLM, which ponyo has in no venv, while ggrpo.generate
# needs only transformers 5.17, which the overlay venv has. E2 therefore runs alone on totoro GPU1 for ~18 h, and
# rather than leave this card idle that whole time it produces the arms' E1 halves. Each half is useless until GPU1
# frees up and runs the other 192 tasks, so these are unfinished arms by design -- kill this lane without loss if the
# card is wanted for something that finishes.
# Started by ops/lanes/ship_lr_grid.sh once the three arms' weights are here.
cd ~/gg/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/gg/ggrpo/src TQDM_DISABLE=1
S=runs/e1_pool_refs.status
echo "WAIT step-368 ponyo half $(date -u +%FT%TZ)" >> $S
until grep -qE "^(LANE_DONE|FAILED|SHORT|RECOVER_FAILED)" runs/codenet_val_grpo368/ponyo.status; do sleep 120; done
for lr in 5e-6 1e-5 2e-5; do
  arm=runs/sft_pool_refs_lr$lr
  O=runs/codenet_val_pool_refs_lr$lr
  [ -s $arm/model.safetensors ] || { echo "NO_WEIGHTS lr=$lr $(date -u +%FT%TZ)" >> $S; exit 1; }
  mkdir -p $O
  cp -n runs/codenet_val_grpo368/pristine_ponyo.jsonl $O/pristine_ponyo.jsonl
  cp -n runs/codenet_val_grpo368/pristine_ponyo.jsonl $O/val_ponyo.jsonl
  echo "E1_START lr=$lr $(date -u +%FT%TZ)" >> $S
  runs/codenet_val_beam8_lane.sh ponyo "$HOME/gg/ggrpo" "$HOME/gg/ggrpo-pilot/.venv-beam/bin/python" $arm $O
  echo "E1_EXIT $? lr=$lr $(date -u +%FT%TZ)" >> $S
done
echo "E1_DONE $(date -u +%FT%TZ)" >> $S
