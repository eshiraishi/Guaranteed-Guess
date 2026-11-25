#!/usr/bin/env bash
# totoro GPU0 (Blackwell): val E2 (16 samples x 383 tasks) for every checkpoint the action plan's measurement phase
# needs (Obsidian "GG RLVR action plan after the first runs", Enzo, Nov 21: "Ok"). Chained behind the SFT-on-pool-
# references grid (e2_lr_grid_gpu0.sh), on the same card and the same two-shard sampling, so every E2 in the plan is
# one GPU type and one lane. Validation only.
#
# Models, in order of what the decisions need first: the SFT start; GRPO seeds 0 and 1 and online RFT seed 0 at the ends
# of pass 2 and pass 1; the permuted-reward control at 184; the start at lower temperatures (is GRPO's gain more than
# sharpening?); then steps 92 and 276 for the curves.
# A finished model (6,128 rows) is skipped, so the lane can be restarted. A failed model stops the lane.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
S=runs/e2_phase1.status
O=runs/e2_phase1
mkdir -p $O
echo "WAIT pool-refs grid $(date -u +%FT%TZ)" >> $S
# Only lines after the grid's last RESTART count: the file keeps an E2_DONE from the Nov 20 03:21Z crash.
# Nov 21, 06:41Z: the grid's three E2 outputs were checked complete; --no-wait starts without the grid wait.
[ "${1:-}" = "--no-wait" ] || until awk '/^RESTART/{d=0} /^E2_DONE/{d=1} END{exit !d}' runs/e2_pool_refs.status; do
  if awk '/^RESTART/{f=0} /^E2_EXIT [1-9]/{f=1} END{exit !f}' runs/e2_pool_refs.status; then
    echo "STOP the pool-refs grid failed $(date -u +%FT%TZ)" >> $S; exit 1
  fi
  sleep 120
done

sample_two_shards() {   # as e2_two_shards.sh, plus the temperature
  local name=$1 model=$2 temperature=$3 out=$O/$1_sample16.jsonl log=$O/$1.log pids=() code=0
  for shard in 0 1; do
    .venv/bin/python -m ggrpo.sample --sample.model $model --sample.dtype bfloat16 \
      --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples 16 --sample.seed codenet-val-$name \
      --sample.temperature $temperature --sample.stop_mixed 100000 --sample.gpu_memory_utilization 0.42 \
      --sample.shard $shard --sample.out $out < /dev/null >> $log.shard$shard 2>&1 &
    pids+=($!)
    sleep 180   # vLLM sizes its KV cache from the free memory at start: let shard 0 finish profiling first
  done
  # Success is the output, not each shard's exit code: on Nov 21 one shard died at vLLM startup (GPU memory held by an
  # orphaned engine) while the other scored every task, and the exit code alone marked a complete E2 as failed.
  for pid in "${pids[@]}"; do wait $pid || { code=$?; echo "SHARD_EXIT $code pid=$pid $name $(date -u +%FT%TZ)" >> $S; }; done
  .venv/bin/python - $out <<'PY'
import sys
import polars as pl
rows = pl.read_ndjson(sys.argv[1])
unique = rows.select("task_id", "sample").unique().height
assert rows.height == unique == 6128, f"{rows.height} rows, {unique} unique (task_id, sample), want 6128"
PY
  local result=$?
  [ $result -eq 0 ] || echo "INCOMPLETE $out" >> $S
  return $result
}

while read -r name model temperature; do
  out=$O/${name}_sample16.jsonl
  if [ -s $out ] && [ "$(wc -l < $out)" -eq 6128 ]; then echo "SKIP $name done" >> $S; continue; fi
  [ -s $model/model.safetensors ] || { echo "NO_WEIGHTS $name $model $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "E2_START $name $model T=$temperature $(date -u +%FT%TZ)" >> $S
  sample_two_shards $name $model $temperature
  code=$?
  echo "E2_EXIT $code $name $(date -u +%FT%TZ)" >> $S
  [ "$code" = 0 ] || exit 1
done <<'MODELS'
start_T1.0 runs/sft_full_seed0 1.0
grpo_seed0_368 runs/codenet_seed0/checkpoint-368 1.0
grpo_seed1_368 runs/codenet_seed1/checkpoint-368 1.0
grpo_seed0_184 runs/codenet_seed0/checkpoint-184 1.0
grpo_seed1_184 runs/codenet_seed1/checkpoint-184 1.0
rft_seed0_368 runs/rft_seed0/checkpoint-368 1.0
rft_seed0_184 runs/rft_seed0/checkpoint-184 1.0
permuted_seed0_184 runs/codenet_random_seed0/checkpoint-184 1.0
start_T0.8 runs/sft_full_seed0 0.8
start_T0.6 runs/sft_full_seed0 0.6
start_T0.4 runs/sft_full_seed0 0.4
start_T0.2 runs/sft_full_seed0 0.2
grpo_seed0_92 runs/codenet_seed0/checkpoint-92 1.0
grpo_seed1_92 runs/codenet_seed1/checkpoint-92 1.0
rft_seed0_92 runs/rft_seed0/checkpoint-92 1.0
permuted_seed0_92 runs/codenet_random_seed0/checkpoint-92 1.0
grpo_seed0_276 runs/codenet_seed0/checkpoint-276 1.0
grpo_seed1_276 runs/codenet_seed1/checkpoint-276 1.0
rft_seed0_276 runs/rft_seed0/checkpoint-276 1.0
MODELS
echo "E2_DONE $(date -u +%FT%TZ)" >> $S
