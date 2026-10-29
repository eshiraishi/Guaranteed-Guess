#!/usr/bin/env bash
# SFT baseline on totoro GPU0 (sft.yaml): GG's checkpoint fine-tuned on the GRPO train problems with GG's recipe.
# Deployed as ~/ggrpo/runs/sft.sh; chain_arms.sh runs it in the foreground with no arguments and waits for
# "SFT_EXIT 0 " in runs/sft.status. Needs ~/ggrpo/sft.yaml and data/sft_train.jsonl (python -m ggrpo.data on the Mac).
# Saves runs/sft_seedN/checkpoint-{54,108}; they are saved by transformers 5.x, so 4.x stacks (BringUp) need the RoPE
# view dir, as for the GRPO checkpoints.
# usage: sft.sh [SEED]   (default 0)
cd ~/ggrpo
seed=${1:-0} name=sft_seed${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=sft-seed$seed WANDB_RESUME=allow   # W&B as full_seed.sh: fixed id, a relaunch continues the run
set -a; . ./.env; set +a
echo "SFT_START seed=$seed $(date)" >> runs/sft.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 > runs/${name}_gpu.csv &
SMI=$!
.venv/bin/trl sft --config sft.yaml --seed $seed --output_dir runs/$name > runs/$name.log 2>&1
rc=$?
kill $SMI
echo "SFT_EXIT $rc $(date)" >> runs/sft.status
