#!/usr/bin/env bash
# The single full-parameter GRPO run on GPU0: grpo_codenet.yaml through ggrpo.record, which saves every rollout and
# reseeds vLLM's sampling (688e689; colocate vLLM otherwise resets the global RNG to 0 for every seed).
#
# WANDB_API_KEY comes from .env, exactly as full_seed.sh and sft_full.sh load it: uv is not on PATH in a
# non-interactive ssh, so bash sources it. Without this line the run dies in 38 s on wandb's "No API key configured",
# which is how the memory probe died on Nov 16 -- grpo_codenet.yaml is report_to [wandb] and nothing else exports the
# key. The run id grpo-codenet-seedN is fixed and readable (no plan IDs on W&B, Enzo, Nov 10), so a relaunch continues
# the same run; a relaunch also resumes from the newest checkpoint in the output folder.
#
# The pool must exist first (ggrpo.pool writes data/rl/pool.jsonl from the census): the lane refuses to start without
# it rather than letting TRL read an empty dataset.
# usage: grpo_codenet.sh SEED [ARM]   (ARM: grpo, the default, or rft for the online-RFT control)
set -uo pipefail
cd ~/ggrpo
seed=$1 arm=${2:-grpo}
case $arm in                            # the online-RFT control is the same lane with one module swapped (prereg.md: GRPO's sampler, LR and steps)
  grpo) module=ggrpo.record;   name=codenet_seed$seed; run="GRPO (datasets: mbpp+codenet)" ;;
  rft)  module=ggrpo.positive; name=rft_seed$seed;     run="Online RFT (datasets: mbpp+codenet)" ;;
  *)    echo "unknown arm $arm (grpo|rft)" >&2; exit 1 ;;
esac
[ -s data/rl/pool.jsonl ] || { echo "no data/rl/pool.jsonl: run ggrpo.pool first" >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
export WANDB_RUN_ID=$arm-codenet-seed$seed WANDB_RESUME=allow
set -a; . ./.env; set +a
resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "GRPO_START arm=$arm seed=$seed resume=${resume:-none} $(date -u +%FT%TZ)" >> runs/$name.status
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> runs/${name}_gpu.csv &
SMI=$!
.venv/bin/python -m $module --config grpo_codenet.yaml --seed $seed --output_dir runs/$name --run_name "$run" \
  ${resume:+--resume_from_checkpoint $resume} >> runs/$name.log 2>&1
echo "GRPO_EXIT $? $(date -u +%FT%TZ)" >> runs/$name.status
kill $SMI
