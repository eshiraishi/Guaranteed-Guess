#!/usr/bin/env bash
# The pass-rate filter over every RL task (Enzo, Nov 17: "All 7,278").
#
# Replaces census.sh's sampling step, which stopped at sample.stop_mixed 960 (pool.cap 864 plus margin). The pool is a
# prefix -- ggrpo.pool takes mixed.head(cap) in census rank order -- so sampling past 960 cannot change which 864 tasks
# the planned run trains on. It costs GPU0 time and buys two things: the pool size never has to be reopened if the
# budget conversation goes well, and the mixed rate is then measured on all 7,278 tasks rather than a 383-task slice.
# stop_mixed is 100000 in config.toml, above any reachable count, so the loop ends on "no task left to claim".
#
# Resumes: claim_chunk skips every task already scored in sample.out and every task another shard has claimed, so a
# restart continues the same file. Everything upstream (SFT_EXIT 0, the fp16 export) is already done, so this waits on
# nothing. SHARD_EXIT goes to its own status file; runs/census.status keeps the stopped lane's history.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
S=runs/census_full.status
echo "START $(date -u +%FT%TZ)" >> $S
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m ggrpo.sample --sample.shard gpu0 --sample.chunk 64 >> runs/census_gpu0.log 2>&1
echo "SHARD_EXIT $? $(date -u +%FT%TZ)" >> $S
