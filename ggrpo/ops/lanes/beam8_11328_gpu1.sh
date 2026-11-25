#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER), under runs/gpu1.lock: the start check's 192-task half (beam 8, CodeNet val) on the
# finished full-data SFT model, then that model's sampled eval (8 samples, monitoring, the last point of the curve).
# Waits for the SFT's clean exit and for save_model's output. Enzo, Nov 16: the filter runs on GPU0 alone so this card
# is free for the start check, which cannot move to the Blackwell (GG's 65/383 was measured on the 4070s).
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
O=runs/codenet_val_11328; S=$O/gpu1.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^SFT_EXIT" runs/sft_full_seed0.status; do sleep 120; done
grep -q "^SFT_EXIT 0" runs/sft_full_seed0.status || { echo "SFT_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
until [ -s runs/sft_full_seed0/model.safetensors ]; do sleep 30; done
sleep 60   # save_model writes the weights before the tokenizer files; do not read the folder mid-write
flock runs/gpu1.lock $O/lane.sh gpu1 "$HOME/ggrpo" "$HOME/ggrpo/.venv/bin/python"
echo "SAMPLED_START $(date -u +%FT%TZ)" >> runs/codenet_val_sft_full_11328.status
flock runs/gpu1.lock .venv/bin/python -m ggrpo.sample --sample.model runs/sft_full_seed0 \
  --sample.dtype bfloat16 --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples 8 \
  --sample.seed codenet-val-sft-full-11328 --sample.stop_mixed 100000 \
  --sample.out runs/sft_full_11328_codenet_val_sample8.jsonl >> runs/codenet_val_sft_full_11328.log 2>&1
echo "SAMPLED_DONE $? $(date -u +%FT%TZ)" >> runs/codenet_val_sft_full_11328.status
