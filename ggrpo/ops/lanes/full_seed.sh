#!/usr/bin/env bash
# Another seed of the full-parameter GRPO run on GPU0: grpo.yaml unchanged except seed and output_dir
# (TRL's parser lets command-line args override the config file). Seed 0 is runs/full (Oct 22-23).
# W&B (Enzo, Oct 23): every seed is named "GRPO" (grpo.yaml run_name); the seed is in the run config. The run id
# grpo-seedN is fixed, so a relaunch continues the same run. WANDB_API_KEY comes from .env (gitignored; uv is
# not on PATH in non-interactive ssh, so bash exports it).
# Seed 0 and seed 1's steps 1-27 trained on Trackio ("baseline, seed=N"); ops/migrate/trackio_to_wandb.py copies them.
# usage: full_seed.sh SEED
cd ~/ggrpo
seed=$1 name=full_seed$1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=grpo-seed$seed WANDB_RESUME=allow
set -a; . ./.env; set +a
echo "FULL_START seed=$seed $(date)" >> runs/$name.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 > runs/${name}_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config grpo.yaml --seed $seed --output_dir runs/$name > runs/$name.log 2>&1
echo "FULL_EXIT $? $(date)" >> runs/$name.status
kill $SMI
