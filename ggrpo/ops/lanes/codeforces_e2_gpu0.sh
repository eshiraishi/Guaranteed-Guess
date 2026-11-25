#!/usr/bin/env bash
# totoro GPU0 (Blackwell): Codeforces val E2 (16 samples x 358 tasks, T 1.0) of the start, GRPO seed 0 step 184 and
# GRPO seed 1 step 368 (Codeforces thread's run list, Nov 22; Enzo, Nov 25: "Ok to run on gpu0 for consistency", so
# every E2 is on the Blackwell). Same sampler settings as phase2_gpu0.sh; unscored, the Mac scores the files
# (e2_score_mac.sh). Validation only: never data/codeforces/test.jsonl. A complete file is skipped; any failure stops.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
S=runs/codeforces_e2.status
O=runs/e2_codeforces
T=runs/codeforces_val_tasks_358.jsonl   # Nov 25: the 359 minus codeforces/611B/15115815 (53,767 prompt tokens > the 32,768 context)
mkdir -p $O

stop() { echo "STOP $* $(date -u +%FT%TZ)" >> $S; exit 1; }
[ "$(wc -l < $T)" -eq 358 ] || stop "$T is not the 358 val tasks"
[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] || stop "GPU0 busy"
for job in "start runs/sft_full_seed0" "grpo_seed0_184 runs/codenet_seed0/checkpoint-184" \
           "grpo_seed1_368 runs/codenet_seed1/checkpoint-368"; do
  set -- $job
  name=$1 model=$2 out=$O/$1_sample16.jsonl pids=()
  if [ -s $out ] && [ "$(wc -l < $out)" -eq $((358 * 16)) ]; then echo "SKIP $name done" >> $S; continue; fi
  [ -s $model/model.safetensors ] || stop "NO_WEIGHTS $name $model"
  echo "E2_START $name $model $(date -u +%FT%TZ)" >> $S
  for shard in 0 1; do
    .venv/bin/python -m ggrpo.sample --sample.model $model --sample.dtype bfloat16 \
      --sample.tasks $T --sample.samples 16 --sample.seed codeforces-val-$name \
      --sample.temperature 1.0 --sample.score false --sample.stop_mixed null --sample.gpu_memory_utilization 0.42 \
      --sample.shard $shard --sample.out $out < /dev/null >> $O/$name.log.shard$shard 2>&1 &
    pids+=($!)
    sleep 180
  done
  for pid in "${pids[@]}"; do wait $pid || stop "SHARD_EXIT pid=$pid $name"; done
  .venv/bin/python - $out <<'PY' || stop "incomplete $name"
import sys
import polars as pl
rows = pl.read_ndjson(sys.argv[1])
unique = rows.select("task_id", "sample").unique().height
assert rows.height == unique == 358 * 16, f"{rows.height} rows, {unique} unique (task_id, sample), want {358 * 16}"
PY
  echo "E2_DONE $name $(date -u +%FT%TZ)" >> $S
  for i in $(seq 20); do [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 30; done
done
echo "ALL_DONE $(date -u +%FT%TZ)" >> $S
