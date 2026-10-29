#!/usr/bin/env bash
# Full-FT probe: 2 steps with the real grpo.yaml, GPU0 memory every 15 s. Waits until the QLoRA smoke queue is gone.
cd ~/ggrpo
while pgrep -f "smoke_hf.yaml|after_smoke.sh|smoke_hf.sh" >/dev/null; do sleep 30; done
echo "PROBE_START $(date)" >> runs/probe_full.status
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 15 > runs/probe_full_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config grpo.yaml --max_steps 2 --save_steps 2 --report_to none \
  --output_dir runs/probe_full --run_name probe_full > runs/probe_full.log 2>&1
echo "PROBE_EXIT $? $(date)" >> runs/probe_full.status
kill $SMI
