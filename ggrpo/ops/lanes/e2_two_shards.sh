#!/usr/bin/env bash
# One E2 (16 samples x 383 val tasks) on totoro GPU0 as two ggrpo.sample shards on the same card and output file.
# Each round of sample.py generates a chunk, then waits for the reward VM to score it; on the Blackwell that wait was
# about half of each round (Nov 20: 11-17 min generating, 1-20 min scoring), so a second shard generates while the
# first is scored. The shards split the tasks through sample.py's claim files; the samples of each task do not depend on
# which shard drew them beyond vLLM's sampling noise, and every model in a comparison goes through this same lane.
# usage: e2_two_shards.sh MODEL SEED_SALT OUT LOG
set -uo pipefail
cd ~/ggrpo
model=$1 salt=$2 out=$3 log=$4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
pids=()
for shard in 0 1; do
  .venv/bin/python -m ggrpo.sample --sample.model $model --sample.dtype bfloat16 \
    --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples 16 --sample.seed $salt \
    --sample.stop_mixed 100000 --sample.gpu_memory_utilization 0.42 --sample.shard $shard \
    --sample.out $out >> $log.shard$shard 2>&1 &
  pids+=($!)
  sleep 180   # vLLM sizes its KV cache from the free memory at start: let shard 0 finish profiling first
done
code=0
# Success is the output, not each shard's exit code: on Nov 21 one shard died at vLLM startup (GPU memory held by an
# orphaned engine) while the other scored every task, and the exit code alone marked a complete E2 as failed.
for pid in "${pids[@]}"; do wait $pid || { code=$?; echo "SHARD_EXIT $code pid=$pid $out" >&2; }; done
.venv/bin/python - $out <<'PY'
import sys
import polars as pl
rows = pl.read_ndjson(sys.argv[1])
unique = rows.select("task_id", "sample").unique().height
assert rows.height == unique == 6128, f"{rows.height} rows, {unique} unique (task_id, sample), want 6128"
PY
