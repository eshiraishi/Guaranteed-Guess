#!/usr/bin/env bash
set -euo pipefail
cd ~/cs8001-step0
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off
pip install -q torch transformers accelerate 2>&1 | tail -1
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
python gen.py tasks_O2.jsonl bench.jsonl --device cuda --max_new_tokens 4096 --ids humaneval/41,humaneval/63,humaneval/37
echo BENCH_DONE
