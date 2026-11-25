#!/usr/bin/env bash
# GPU0, between the fp16 export and the census: two GRPO updates on the 16 longest RL prompts, with the real reward.
# grpo_codenet.yaml's per_device_train_batch_size 1 has never been run on a 32k-token batch with this start model
# ("UNTESTED: memory pilot on the longest batch first"), and an OOM a day into the single run would cost the run.
# Enzo, Nov 16: go slowly and keep concrete results, so prove the worst batch fits before anything long starts.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
S=runs/probe_memory.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until [ -s runs/start_fp16/model.safetensors ] || [ -s runs/start_fp16/model.safetensors.index.json ]; do sleep 60; done
sleep 30
echo "START $(date -u +%FT%TZ)" >> $S
.venv/bin/python -m ggrpo.record --config runs/probe_memory.yaml > runs/probe_memory.log 2>&1
code=$?
echo "EXIT $code $(date -u +%FT%TZ)" >> $S
# the step time of these two updates is on the longest prompts, so it is an upper bound, not the run's rate
tr '\r' '\n' < runs/probe_memory.log | grep -E "s/it|train_runtime|OutOfMemory" | tail -4 >> $S
rm -rf runs/probe_memory   # weights discarded; the status file and the log are the result
