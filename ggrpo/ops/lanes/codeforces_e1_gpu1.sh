#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER): Codeforces val E1 (beam 8, 358 tasks) of the start, GRPO seed 0 step 184 and GRPO seed 1
# step 368, one model after the other, all on this card (one GPU type per comparison). Each model runs the whole set as
# one codenet_val_beam8_lane.sh half named gpu1, so out-of-memory tasks are re-generated with the KV cache offloaded
# and nothing is excluded. 358 = the 359 val tasks minus codeforces/611B/15115815 (53,767 prompt tokens > the 32,768
# context; Nov 25). Generation only; the Mac scores. Validation only: never data/codeforces/test.jsonl.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
S=runs/codeforces_e1.status
T=runs/codeforces_val_tasks_358.jsonl
[ "$(wc -l < $T)" -eq 358 ] || { echo "STOP $T is not the 358 val tasks $(date -u +%FT%TZ)" >> $S; exit 1; }
for job in "start runs/sft_full_seed0" "grpo_seed0_184 runs/codenet_seed0/checkpoint-184" \
           "grpo_seed1_368 runs/codenet_seed1/checkpoint-368"; do
  set -- $job
  name=$1 model=$2 O=runs/e1_codeforces/$1
  if grep -q "^LANE_DONE" $O/gpu1.status 2>/dev/null; then echo "SKIP $name done" >> $S; continue; fi
  [ -s $model/model.safetensors ] || { echo "STOP NO_WEIGHTS $name $model $(date -u +%FT%TZ)" >> $S; exit 1; }
  mkdir -p $O
  [ -s $O/pristine_gpu1.jsonl ] || { cp $T $O/pristine_gpu1.jsonl; cp $T $O/val_gpu1.jsonl; }
  echo "E1_START $name $model $(date -u +%FT%TZ)" >> $S
  flock runs/gpu1.lock runs/codenet_val_beam8_lane.sh gpu1 "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python" $model $O \
    || { echo "STOP LANE_FAILED $name $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "E1_DONE $name $(date -u +%FT%TZ)" >> $S
done
echo "ALL_DONE $(date -u +%FT%TZ)" >> $S
