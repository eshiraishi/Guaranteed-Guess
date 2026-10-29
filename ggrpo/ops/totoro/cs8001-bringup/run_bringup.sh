#!/bin/bash
# Both BringUp O2 runs side by side on the RTX PRO 6000 Blackwell (PCI 01:00.0).
cd ~/cs8001-bringup && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
.venv/bin/python -c "
import torch; from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16); flash_attn_func(q, q, q, causal=True); torch.cuda.synchronize()
print('FA2 OK on', torch.cuda.get_device_name(0))" 2>&1 | tee -a run.log || exit 1
.venv/bin/python gen_bringup.py gg-bench-bringup-O2.parquet bringup_O2_released.jsonl --mode released 2>&1 | grep -v -i warn >> released.log &
.venv/bin/python gen_bringup.py gg-bench-bringup-O2.parquet bringup_O2_beam8.jsonl --mode beam8 2>&1 | grep -v -i warn >> beam8.log &
wait; echo BRINGUP_DONE $(date -u) | tee -a run.log
