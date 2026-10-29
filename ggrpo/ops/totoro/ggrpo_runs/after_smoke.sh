#!/usr/bin/env bash
# After the no-vLLM smokes: chunked-prefill equivalence check, then resume BringUp beam8 on the free GPU.
cd ~/ggrpo
until grep -q QLORA_HF_EXIT runs/smoke_hf.status; do sleep 30; done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python runs/prefill_chunk_check.py > runs/prefill_chunk_check.log 2>&1
echo "CHECK_EXIT $? $(date)" >> runs/smoke_hf.status
cd ~/cs8001-bringup && echo "BEAM8_RESUMED $(date)" >> run.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python gen_bringup.py gg-bench-bringup-O2.parquet bringup_O2_beam8.jsonl --mode beam8 >> beam8_resume.log 2>&1
echo "BEAM8_DONE $(date)" >> run.log
