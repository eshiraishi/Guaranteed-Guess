#!/usr/bin/env bash
set -euo pipefail
cd ~/cs8001-step0
source venv/bin/activate
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off
python gen.py tasks_O2.jsonl preds_full_raw.jsonl --device cuda --max_new_tokens 16000
echo FULL_DONE
