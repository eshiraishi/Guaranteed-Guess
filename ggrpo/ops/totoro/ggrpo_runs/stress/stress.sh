#!/usr/bin/env bash
# Worst-case memory test for micro-batch 4 x 32: one GRPO step on the 16 longest train prompts (10,749 to 4,136 tokens).
# Starts once the reward tunnel answers on totoro.
cd ~/ggrpo
until curl -sf -m 5 -o /dev/null http://127.0.0.1:8765/docs; do sleep 20; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "STRESS_START $(date)" >> runs/stress/status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used --format=csv,noheader -lms 1000 > runs/stress/gpu.csv &
SMI=$!
.venv/bin/trl grpo --config runs/stress/grpo.yaml --max_steps 1 --report_to none --output_dir runs/stress/out --save_strategy no --log_completions false > runs/stress/log 2>&1
echo "STRESS_EXIT $? $(date)" >> runs/stress/status
kill $SMI
