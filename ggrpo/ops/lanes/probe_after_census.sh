#!/usr/bin/env bash
# The memory probe, chained after the census instead of before it (Enzo, Nov 17).
#
# census.sh ran the probe first, as intended, but it died in 38 s on wandb's "No API key configured": the lane's export
# list carries no WANDB_API_KEY and probe_memory.yaml asked for report_to [wandb]. That is now report_to [], since the
# probe's weights are discarded and its verdict is the step time in its log. It cannot share GPU0 with the census
# (vLLM holds ~84 of 98 GiB), and nothing is lost by waiting: GRPO cannot start before the pool exists either.
#
# Waits for the full census shard (census_full.sh, Nov 17) to exit, then proves a 32k-token batch at per_device_train_batch_size 1 fits on the
# Blackwell before the single real GRPO run commits to it. runs/probe_memory.status carries the verdict.
cd ~/ggrpo
S=runs/probe_after_census.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^SHARD_EXIT" runs/census_full.status; do sleep 120; done
grep -q "^SHARD_EXIT 0" runs/census_full.status || { echo "CENSUS_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
echo "START $(date -u +%FT%TZ)" >> $S
bash runs/probe_memory.sh
echo "PROBE_EXIT $(tail -1 runs/probe_memory.status) $(date -u +%FT%TZ)" >> $S
