#!/usr/bin/env bash
set -euo pipefail
cd ~/cs8001-step0
source venv/bin/activate
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off
python -c "
from transformers import GenerationConfig
g=GenerationConfig.from_pretrained(\"ahmedheakl/gg-armv8-O2\"); print(\"checkpoint generation_config:\", g.repetition_penalty, g.temperature, g.top_k, g.top_p)"
python gen_v2.py tasks_O2.jsonl preds_rep1_raw.jsonl --device cuda --max_new_tokens 16000 --rep 1.0
echo REP1_DONE
