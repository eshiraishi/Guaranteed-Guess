#!/usr/bin/env bash
# CodeNet val (383 tasks) x 8 samples of the full-data SFT at 80% and 90%, filling the two holes in the curve
# (Enzo, Nov 17: GPU1 fell idle when the 11328 eval finished, and option A was to fill them).
#
# config.toml's [track] already declares results/codenet_val/sft_full_{9064,10197}_sample8.scored.jsonl, so these two
# points were always meant to exist: every other decile is measured, and 7931 -> 11328 gained only +0.8 pp, which
# these say was a plateau or a late climb. Monitoring only (prereg section 2), validation only.
#
# Identical to sampled_chain.sh (5665, 6798, 7931) in every argument but the step, so the curve stays one measurement
# repeated: same tasks file, same dtype, same 8 samples, same per-step seed, same lock. Scoring shares the reward VM
# with the census on GPU0 -- the VM was a third loaded when this started, and if the census drops below ~7 tasks/min
# this lane is the one to kill.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
for step in 9064 10197; do
  echo "START $(date -u +%FT%TZ)" >> runs/codenet_val_sft_full_$step.status
  flock runs/gpu1.lock .venv/bin/python -m ggrpo.sample --sample.model runs/sft_full_seed0/checkpoint-$step \
    --sample.dtype bfloat16 --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples 8 \
    --sample.seed codenet-val-sft-full-$step --sample.stop_mixed 100000 \
    --sample.out runs/sft_full_${step}_codenet_val_sample8.jsonl >> runs/codenet_val_sft_full_$step.log 2>&1
  echo "DONE $? $(date -u +%FT%TZ)" >> runs/codenet_val_sft_full_$step.status
done
