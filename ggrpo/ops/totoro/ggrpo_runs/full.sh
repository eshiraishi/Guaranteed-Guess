#!/usr/bin/env bash
# Full-parameter GRPO (grpo.yaml as is) on GPU0. Restart of Oct 22 after the pilot (runs/pilot) OOMed at step 19.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "FULL_START $(date)" >> runs/full.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 > runs/full_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config grpo.yaml > runs/full.log 2>&1
echo "FULL_EXIT $? $(date)" >> runs/full.status
kill $SMI
