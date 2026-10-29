#!/bin/bash
# Full HumanEval-C O2 runs, transformers 4.50.0 + torch 2.6.0 + FA2, cap 32000 - prompt tokens.
cd ~/cs8001-step0/v450 && export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off
.venv/bin/python gen_authors.py gg-bench-armv8-O2.parquet full_greedy_beam.jsonl --mode greedy_beam 2>&1 | grep -v -i warn | tee -a full.log
.venv/bin/python gen_authors.py gg-bench-armv8-O2.parquet full_authors_s0.jsonl --mode authors --seed 0 2>&1 | grep -v -i warn | tee -a full.log
.venv/bin/python gen_authors.py gg-bench-armv8-O2.parquet full_authors_s1.jsonl --mode authors --seed 1 2>&1 | grep -v -i warn | tee -a full.log
echo ALL_DONE | tee -a full.log
