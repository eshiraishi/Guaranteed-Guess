#!/usr/bin/env bash
# totoro GPU0 queue (Enzo, Oct 23): GRPO (DeepSeekMath) seed 0 (running, arm_seed.sh), paused at its next checkpoint
# (54 or 81) for SFT on the same 858 tasks' compiler labels (runs/sft.sh, the data control) so that SFT's evals make the
# midpoint cutoff, then resumed there -> Dr. GRPO seed 0 -> GRPO (TRL default) seed 1 resumed from checkpoint-27 (paused
# there for the arms). TRL 1.16's CLI passes resume_from_checkpoint to trainer.train.
# This version goes live only once ggrpo-dev has confirmed SFT ready and runs/sft.sh is in place. If DSM is past
# checkpoint-81 by then, it does not pause and SFT runs after DSM. SFT always runs before seed 1 resumes.
# The pause: wait until the checkpoint is fully saved (the trainer writes trainer_state.json last), then SIGINT the
# trainer process, as seed 1 was stopped; arm_seed.sh logs ARM_EXIT and later resumes the run in place.
# Each run starts only after the previous one exited 0 and left the GPU; a failed training run stops the queue
# (CHAIN_STOP) so it is looked at before the GPU is reused. A failed SFT (SFT_FAILED) does not hold up the arms, only
# seed 1, until a person removes runs/sft.status to rerun it.
cd ~/ggrpo
S=runs/chain.status
gpu0_free() { until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; }
exited() { tail -n 1 $1 2>/dev/null | grep -qE "(ARM|FULL)_EXIT"; }   # the run's last status line is an exit line
after() {  # after STATUS_FILE: wait for the run's (last) exit line; continue only on exit 0 and an empty GPU0
  until exited $1; do sleep 60; done
  tail -n 1 $1 | grep -qE "(ARM|FULL)_EXIT 0 " || { echo "CHAIN_STOP $1: $(tail -n 1 $1) $(date -u +%FT%TZ)" >> $S; exit 1; }
  gpu0_free
}
sft() {  # run runs/sft.sh (foreground; it writes runs/sft.status: SFT_START / SFT_EXIT) if it is ready and has not run
  [ -f runs/sft.sh ] && [ ! -f runs/sft.status ] || return 0
  echo "START sft $(date -u +%FT%TZ)" >> $S
  bash runs/sft.sh
  rc=$?; grep -q SFT_EXIT runs/sft.status 2>/dev/null || echo "SFT_EXIT $rc (written by chain_arms.sh) $(date)" >> runs/sft.status
  grep -q "SFT_EXIT 0 " runs/sft.status 2>/dev/null || echo "SFT_FAILED $(tail -n 1 runs/sft.status 2>/dev/null) $(date -u +%FT%TZ)" >> $S
  gpu0_free
}
pause_dsm_for_sft() {  # at DSM's next checkpoint (54 or 81) once fully saved: stop DSM, run SFT, resume DSM there
  local n c
  for n in 54 81; do [ -f runs/dsm_seed0/checkpoint-$n/trainer_state.json ] || break; done
  c=runs/dsm_seed0/checkpoint-$n
  [ -f $c/trainer_state.json ] || ! [ -f runs/sft.sh ] || [ -f runs/sft.status ] && return
  until [ -f $c/trainer_state.json ] || exited runs/dsm_seed0.status; do sleep 15; done
  exited runs/dsm_seed0.status && return
  sleep 30   # let the trainer_state.json write finish
  for f in model.safetensors optimizer.pt scheduler.pt rng_state.pth; do [ -s $c/$f ] || return; done
  .venv/bin/python -c "import json, sys; sys.exit(json.load(open('$c/trainer_state.json'))['global_step'] != $n)" || return
  local p=$(pgrep -f "trl/scripts/grpo.py .*--output_dir runs/dsm_seed0( |$)")
  [ "$(echo "$p" | grep -c .)" = 1 ] || { echo "PAUSE_SKIPPED trainer pid not unique: $p $(date -u +%FT%TZ)" >> $S; return; }
  echo "PAUSE dsm_seed0 at checkpoint-$n for sft (SIGINT $p) $(date -u +%FT%TZ)" >> $S
  kill -INT $p
  until exited runs/dsm_seed0.status; do sleep 10; done
  gpu0_free
  sft
  echo "START dsm_seed0 resume from checkpoint-$n $(date -u +%FT%TZ)" >> $S
  bash runs/arm_seed.sh dsm 0 $n
}
echo "CHAIN_START $(date -u +%FT%TZ)" >> $S
pause_dsm_for_sft
after runs/dsm_seed0.status
sft
echo "START drgrpo_seed0 $(date -u +%FT%TZ)" >> $S
bash runs/arm_seed.sh drgrpo 0
after runs/drgrpo_seed0.status
sft
grep -q "SFT_EXIT 0 " runs/sft.status 2>/dev/null || echo "CHAIN_WAIT sft not done, seed 1 held $(date -u +%FT%TZ)" >> $S
until grep -q "SFT_EXIT 0 " runs/sft.status 2>/dev/null; do sleep 60; sft; done
echo "START full_seed1 resume $(date -u +%FT%TZ)" >> $S
# as full_seed.sh 1, plus the resume; the log and GPU trace are appended, not replaced. Steps 1-27 trained on Trackio
# and are copied to W&B run grpo-seed1 first (ops/migrate/trackio_to_wandb.py), so the resume continues that run
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=grpo-seed1 WANDB_RESUME=allow
set -a; . ./.env; set +a
echo "FULL_RESUME seed=1 from checkpoint-27 $(date)" >> runs/full_seed1.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> runs/full_seed1_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config grpo.yaml --seed 1 --output_dir runs/full_seed1 \
  --resume_from_checkpoint runs/full_seed1/checkpoint-27 >> runs/full_seed1.log 2>&1
echo "FULL_EXIT $? $(date)" >> runs/full_seed1.status
kill $SMI
echo "CHAIN_DONE $(date -u +%FT%TZ)" >> $S
