#!/usr/bin/env bash
# The full-data SFT on GPU0: sft_full.yaml unchanged except seed, through ggrpo.checkpointing in the Liger overlay venv
# (~/ggrpo-pilot/.venv-liger: liger-kernel over .venv's packages; it skips .venv's .pth files, hence PYTHONPATH=src).
# A relaunch resumes from the newest checkpoint in the output folder (full state every 10%) and continues the same W&B
# run: the run id sft-full-seedN is fixed and readable (no plan IDs on W&B, Enzo, Nov 10). WANDB_API_KEY comes from
# .env, as full_seed.sh.
# usage: sft_full.sh SEED
cd ~/ggrpo
seed=$1 name=sft_full_seed$1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=sft-full-seed$seed WANDB_RESUME=allow PYTHONPATH=src
set -a; . ./.env; set +a
resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "SFT_START seed=$seed resume=${resume:-none} $(date)" >> runs/$name.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> runs/${name}_gpu.csv &
SMI=$!
~/ggrpo-pilot/.venv-liger/bin/python -m ggrpo.checkpointing --config sft_full.yaml --seed $seed --output_dir runs/$name \
  ${resume:+--resume_from_checkpoint $resume} >> runs/$name.log 2>&1
echo "SFT_EXIT $? $(date)" >> runs/$name.status
kill $SMI
