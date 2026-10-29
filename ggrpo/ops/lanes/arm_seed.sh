#!/usr/bin/env bash
# Baseline arms (Enzo, Oct 23): grpo.yaml unchanged except each paper's algorithm switches, seed, output_dir and run_name.
# dsm = GRPO (DeepSeekMath 2402.03300): per-sequence loss, KL 0.04, truncated completions kept in the loss.
# drgrpo = Dr. GRPO (2503.20783): loss / constant max_completion_length, no std scaling, truncated completions kept.
# With CHECKPOINT (e.g. 54), resumes runs/NAME/checkpoint-CHECKPOINT in place (DSM was paused there for SFT, Enzo Oct 23);
# the status, log and GPU trace are appended to, never replaced.
# usage: arm_seed.sh ARM SEED [CHECKPOINT]
cd ~/ggrpo
# Starts only from chain_arms.sh (or by hand with GG_ARM_MANUAL=1): the arms tmux of 11:14 GST Oct 23 runs
# "arm_seed.sh drgrpo 0" right after DSM exits, which would race the queue for GPU0 and Dr. GRPO's output dir.
ps -o args= -p $PPID | grep -q chain_arms.sh || [ -n "$GG_ARM_MANUAL" ] ||
  { echo "REFUSED arm_seed.sh $* (not started by chain_arms.sh) $(date -u +%FT%TZ)" >> runs/chain.status; exit 3; }
arm=$1 seed=$2 ck=$3 name=$1_seed$2
case $arm in
  dsm) label="GRPO (DeepSeekMath)"; set -- --loss_type grpo --beta 0.04 --mask_truncated_completions false ;;
  drgrpo) label="Dr. GRPO"; set -- --loss_type dr_grpo --scale_rewards none --mask_truncated_completions false ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
[ -n "$ck" ] && set -- "$@" --resume_from_checkpoint runs/$name/checkpoint-$ck
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# W&B as full_seed.sh: the name is the recipe, the seed is in the run config, the id is fixed (a resume continues the run)
export WANDB_RUN_ID=$arm-seed$seed WANDB_RESUME=allow
set -a; . ./.env; set +a
echo "ARM_START $arm seed=$seed${ck:+ resumed from checkpoint-$ck} $(date)" >> runs/$name.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> runs/${name}_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config grpo.yaml "$@" --seed $seed --output_dir runs/$name --run_name "$label" >> runs/$name.log 2>&1
echo "ARM_EXIT $? $(date)" >> runs/$name.status
kill $SMI
