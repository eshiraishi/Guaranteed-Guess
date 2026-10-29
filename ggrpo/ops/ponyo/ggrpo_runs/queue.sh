#!/usr/bin/env bash
# GPU queue (one 16 GB GPU): starts after the peer session frees it.
set -u
cd ~/gg/ggrpo
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off
until grep -q GREEDY_DONE_GPU_FREED ~/cs8001-step0/v450/full.log 2>/dev/null; do sleep 60; done
echo "QUEUE_START $(date -u +%FT%TZ)"
paper() { ~/.local/bin/uv run --extra paper python -m "$@"; }
modern() { UV_PROJECT_ENVIRONMENT=.venv-train ~/.local/bin/uv run --extra train python -m "$@"; }
paper ggrpo.paper 2>&1 | tee runs/paper_as_released.log
paper ggrpo.equivalence --equivalence.attn_implementation flash_attention_2 --equivalence.out runs/equivalence/paper_flash.jsonl 2>&1 | tee runs/equivalence_paper_flash.log
paper ggrpo.equivalence --equivalence.attn_implementation sdpa --equivalence.out runs/equivalence/paper_sdpa.jsonl 2>&1 | tee runs/equivalence_paper_sdpa.log
modern ggrpo.equivalence --equivalence.attn_implementation sdpa --equivalence.out runs/equivalence/modern_sdpa.jsonl 2>&1 | tee runs/equivalence_modern_sdpa.log
# modern ggrpo.generate moved to totoro GPU 1 (03:00.0), Oct 22
echo "QUEUE_DONE $(date -u +%FT%TZ)"
