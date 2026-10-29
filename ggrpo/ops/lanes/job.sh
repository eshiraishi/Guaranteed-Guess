#!/bin/bash
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -x ws-l1-[001-021],ws-l2-[001-021],ws-l3-[001-021],ws-l5-[001-032]
#SBATCH -o logs/%x-%j.log
# usage: sbatch -J name job.sh OUT.jsonl MODE FILES [--model DIR]   (one GPU type per run: gen_bringup.py refuses any other card)
cd ~/cs8001-bringup
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
export HF_HUB_OFFLINE=1   # model is pre-downloaded on the login node
exec .venv/bin/python gen_bringup.py gg-bench-bringup-O2.parquet "$1" --mode "$2" --gpu "RTX 5000 Ada" --files "$3" "${@:4}"
