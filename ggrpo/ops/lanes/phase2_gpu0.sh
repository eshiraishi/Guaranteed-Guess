#!/usr/bin/env bash
# totoro GPU0 (Blackwell): the action plan's Phase 2 development grid (Obsidian "GG RLVR action plan after the first
# runs"; Enzo, Nov 21: "yes you can launch"), chained behind Phase 1's E2 lane. Validation only.
#
# Each training run is followed by val E2 (16 samples x 383 tasks, T 1.0) of its 1- and 2-pass models, on the same
# two-shard lane as Phase 1, so every E2 in the plan is one GPU type. The rules are docs/paper-plan/current/
# selection-rules.md (committed b4344f8, Nov 21, before any Phase 2 run): the same LR grid and horizons {1, 2 passes}
# for every trained arm; the SFT arms get separate 1- and 2-epoch runs, since under cosine a mid-run checkpoint is not a
# shorter run. Second seeds, grid-edge LRs and the temperature sweep depend on these results and are queued after them.
# The jobs are runs/phase2_jobs.txt, one call per line, re-read after every job, so a job can be inserted or reordered
# without stopping the one running (Enzo, Nov 21: "GRPO first": the GRPO grid, data-breadth check and GRPO follow-ups
# come before the control arms). A job is done when its JOB_DONE line is in the status file.
# E2 runs with --sample.score false (Nov 23): it only generates, and the Mac scores the files (ggrpo.score_samples),
# so no eval waits on the reward tunnel. Only the online runs need the reward server.
# Online runs are one 368-step run each, not 184 + resume: the LR is constant after warmup, so both give the same schedule.
# Restartable: a training run that logged TRAIN_EXIT 0 is skipped, an unfinished one resumes from its newest checkpoint;
# a complete E2 (6,128 unique rows) is skipped. Any failure stops the lane.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
set -a; . ./.env; set +a
S=runs/phase2.status
O=runs/e2_phase2
mkdir -p $O

stop() { echo "STOP $* $(date -u +%FT%TZ)" >> $S; exit 1; }
gpu0_free() {   # an orphaned vLLM engine can outlive its parent (Nov 21): wait up to 10 min, then stop
  for i in $(seq 20); do [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] && return; sleep 30; done
  stop "GPU0 still busy: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' ')"
}
disk_ok() { [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -ge 100 ] || stop "under 100 GB free"; }

sample_two_shards() {   # as e2_phase1_gpu0.sh, plus temperature and sample count
  local name=$1 model=$2 temperature=$3 samples=$4 out=$O/$1_sample16.jsonl log=$O/$1.log pids=()
  for shard in 0 1; do
    .venv/bin/python -m ggrpo.sample --sample.model $model --sample.dtype bfloat16 \
      --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples $samples --sample.seed codenet-val-$name \
      --sample.temperature $temperature --sample.score false --sample.stop_mixed null --sample.gpu_memory_utilization 0.42 \
      --sample.shard $shard --sample.out $out < /dev/null >> $log.shard$shard 2>&1 &
    pids+=($!)
    sleep 180
  done
  for pid in "${pids[@]}"; do wait $pid || echo "SHARD_EXIT $? pid=$pid $name $(date -u +%FT%TZ)" >> $S; done
  .venv/bin/python - $out $samples <<'PY'
import sys
import polars as pl
rows = pl.read_ndjson(sys.argv[1])
want = 383 * int(sys.argv[2])
unique = rows.select("task_id", "sample").unique().height
assert rows.height == unique == want, f"{rows.height} rows, {unique} unique (task_id, sample), want {want}"
PY
}
e2() {   # e2 NAME MODEL_DIR [TEMPERATURE SAMPLES]
  local out=$O/$1_sample16.jsonl temperature=${3:-1.0} samples=${4:-16}
  if [ -s $out ] && [ "$(wc -l < $out)" -eq $((383 * samples)) ]; then echo "SKIP e2 $1 done" >> $S; return; fi
  [ -s $2/model.safetensors ] || stop "NO_WEIGHTS $1 $2"
  gpu0_free
  echo "E2_START $1 $2 T=$temperature n=$samples $(date -u +%FT%TZ)" >> $S
  sample_two_shards $1 $2 $temperature $samples
  local code=$?
  echo "E2_EXIT $code $1 $(date -u +%FT%TZ)" >> $S
  [ $code -eq 0 ] || stop "e2 $1"
}
trained() { grep -q "^TRAIN_EXIT 0 $1 " $S 2>/dev/null; }
train() {   # train NAME WANDB_ID COMMAND...
  local name=$1 id=$2; shift 2
  trained $name && { echo "SKIP train $name done" >> $S; return; }
  gpu0_free; disk_ok
  export WANDB_RUN_ID=$id WANDB_RESUME=allow
  echo "TRAIN_START $name $(date -u +%FT%TZ)" >> $S
  "$@" >> runs/$name.log 2>&1
  local code=$?
  echo "TRAIN_EXIT $code $name $(date -u +%FT%TZ)" >> $S
  [ $code -eq 0 ] || stop "train $name"
}
online() {   # online ARM LR SEED CONFIG STEPS LABEL: GRPO (ggrpo.record) or online RFT (ggrpo.positive)
  local arm=$1 lr=$2 seed=$3 config=$4 steps=$5 label=$6 module name resume
  case $arm in grpo) module=ggrpo.record ;; rft) module=ggrpo.positive ;; esac
  name=${arm}_${label}_seed$seed
  resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
  train $name $arm-codenet-$label-seed$seed .venv/bin/python -m $module --config $config --seed $seed \
    --learning_rate $lr --max_steps $steps --output_dir runs/$name \
    --run_name "$( [ $arm = grpo ] && echo GRPO || echo Online RFT ) (datasets: mbpp+codenet, $label)" \
    ${resume:+--resume_from_checkpoint $resume}
  for step in $((steps / 2)) $steps; do e2 ${name}_$step runs/$name/checkpoint-$step; done   # 1 and 2 passes
}
sft() {   # sft DATASET TAG LR EPOCHS: from the SFT start, final model only
  local name=sft_${2}_${4}ep_lr$3
  train $name sft-$2-${4}ep-lr$3 .venv/bin/python -m ggrpo.checkpointing --config sft_references.yaml \
    --dataset_name $1 --learning_rate $3 --num_train_epochs $4 --output_dir runs/$name \
    --run_name "SFT on $2 (lr: $3, epochs: $4)"
  e2 $name runs/$name
}

echo "WAIT phase 1 $(date -u +%FT%TZ)" >> $S
until grep -q "^E2_DONE" runs/e2_phase1.status; do
  grep -q "^STOP" <(tail -1 runs/e2_phase1.status) && stop "phase 1 stopped: $(tail -1 runs/e2_phase1.status)"
  sleep 120
done
echo "PHASE2_START $(date -u +%FT%TZ)" >> $S
while job=$(grep -v -e '^#' -e '^$' runs/phase2_jobs.txt | while read -r line; do
              grep -qxF "JOB_DONE $line" $S || { echo "$line"; break; }; done) && [ -n "$job" ]; do
  case ${job%% *} in e2|online|sft) ;; *) stop "unknown job: $job" ;; esac
  echo "JOB_START $job $(date -u +%FT%TZ)" >> $S
  $job
  echo "JOB_DONE $job" >> $S
done
echo "PHASE2_DONE $(date -u +%FT%TZ)" >> $S
