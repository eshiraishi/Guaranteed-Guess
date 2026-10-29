#!/usr/bin/env bash
# 2-step smokes without vLLM (both arms generate with transformers), to measure step time and memory.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 15 > runs/smoke_hf_gpu.csv &
SMI=$!
.venv/bin/trl grpo --config runs/smoke_hf.yaml --use_vllm false --max_steps 2 --save_steps 2 \
  --output_dir runs/smoke_lora_hf --run_name smoke_lora_hf --report_to none > runs/smoke_lora_hf.log 2>&1
echo "LORA_HF_EXIT $? $(date)" >> runs/smoke_hf.status
.venv/bin/trl grpo --config runs/smoke_hf.yaml --use_vllm false --load_in_4bit true --bnb_4bit_quant_type nf4 --use_bnb_nested_quant true \
  --max_steps 2 --save_steps 2 --output_dir runs/smoke_qlora_hf --run_name smoke_qlora_hf --report_to none > runs/smoke_qlora_hf.log 2>&1
echo "QLORA_HF_EXIT $? $(date)" >> runs/smoke_hf.status
kill $SMI
