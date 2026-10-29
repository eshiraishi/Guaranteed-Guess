#!/usr/bin/env bash
# totoro GPU1, while the eval lane waits for checkpoint-106: sampled dev pass@1 (ops/lanes/sample_eval.py) for the base
# model and the checkpoints given as arguments. Stops before starting a model if checkpoint-106 exists (GPU1 is then
# needed for the headline HumanEval run).
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # ninja for vLLM kernels; FlashInfer cannot JIT its sampler for sm_89 here (plain multinomial is the same distribution at T=1, no top-k/p)
S=runs/sample_gpu1.status; mkdir -p runs/full_eval
for m in "$@"; do
  [ -d runs/full/checkpoint-106 ] && { echo "STOP_FOR_106 $(date -u +%FT%TZ)" >> $S; exit; }
  src=runs/full/checkpoint-$m; [ $m = base ] && src=ahmedheakl/gg-armv8-O2
  echo "SAMPLE_START $m $(date -u +%FT%TZ)" >> $S
  .venv/bin/python runs/sample_eval.py $src data/dev.jsonl runs/full_eval/${m}_dev_sample8.jsonl >> runs/sample_gpu1.log 2>&1
  echo "SAMPLE_DONE $m $? $(date -u +%FT%TZ)" >> $S
done
