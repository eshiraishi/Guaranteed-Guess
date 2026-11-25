#!/usr/bin/env bash
# The pass-rate filter ("census", prereg §2) right after the full-data SFT (Enzo, Nov 13: the Blackwell runs it, since no
# training waits on it). Waits for the SFT's clean exit, exports the fp16 start, then samples on GPU0 (Blackwell) until
# the scored prefix holds sample.stop_mixed mixed tasks. config.toml [sample] otherwise.
# Enzo, Nov 16: GPU0 alone, no GPU1 shard. The start check (beam 8 on the final SFT model) needs the 4070, and it cannot
# move to the Blackwell, since GG's published 65/383 was measured on the 4070s and a comparison uses one GPU type.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
S=runs/census.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^SFT_EXIT" runs/sft_full_seed0.status; do sleep 120; done
grep -q "^SFT_EXIT 0" runs/sft_full_seed0.status || { echo "SFT_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
.venv/bin/python -m ggrpo.fp16 >> runs/census_fp16.log 2>&1 || { echo "FP16_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
# The memory probe first: two GRPO updates on the longest prompts, ~40 min, so the single real run is not the thing
# that discovers a 32k-token batch does not fit. The census does not depend on it (vLLM sampling, not training), so it
# goes ahead either way; runs/probe_memory.status carries the verdict.
bash runs/probe_memory.sh
echo "PROBE_EXIT $(tail -1 runs/probe_memory.status) $(date -u +%FT%TZ)" >> $S
echo "START $(date -u +%FT%TZ)" >> $S
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m ggrpo.sample --sample.shard gpu0 --sample.chunk 64 > runs/census_gpu0.log 2>&1
echo "SHARD_EXIT $? $(date -u +%FT%TZ)" >> $S
