#!/usr/bin/env bash
# totoro GPU1 queue (Nov 22): GRPO seed 0 step 368's ponyo half first, then the RFT beam-8 eval from scratch.
set -uo pipefail
cd ~/ggrpo
runs/beam8_grpo368_ponyohalf_gpu1.sh
O=runs/codenet_val_rft0_368
mkdir -p $O/stopped_20261122 && mv $O/*.log $O/*.status $O/*.attempt $O/beam8_*.jsonl $O/stopped_20261122/ 2>/dev/null
for h in gpu1 ponyo; do cp runs/codenet_val_grpo368/pristine_$h.jsonl $O/pristine_$h.jsonl; cp $O/pristine_$h.jsonl $O/val_$h.jsonl; done
runs/beam8_rft0_gpu1.sh
