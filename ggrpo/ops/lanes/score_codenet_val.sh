#!/usr/bin/env bash
# Runs on the Mac: scores the two complete beam-8 CodeNet-val halves of each arm through the VM's reward server.
# Both arms are 383/383 now that every CUDA-OOM task was re-generated with the KV cache in host RAM. The GG arm is the
# 2026-11-15 rerun of ponyo's half under transformers 5.17 plus totoro's 2026-11-09 half, which was 5.17 already.
set -euo pipefail
cd "$(dirname "$0")/../.."
url=http://192.168.64.2:8765/score
R=results/codenet_val
T=$R/tasks_383.jsonl
S=$R/score.status
echo "SCORE_START $(date -u +%FT%TZ)" >> $S

score() {  # score PREDICTIONS: row and id check, then ggrpo.score next to it
  uv run --frozen python -c "
import polars as pl
p, t = pl.read_ndjson('$1'), pl.read_ndjson('$T')
assert p.height == t.height and set(p['task_id']) == set(t['task_id']), (p.height, t.height)"
  uv run --frozen python -m ggrpo.score --server.url $url --score.tasks $T --score.predictions $1 \
    --score.out ${1%.jsonl}.scored.jsonl --score.reference $R/ref_383.scored.jsonl --log.file runs/ggrpo.jsonl 2>&1 | tail -n 1
}

TOTORO="-o HostKeyAlias=totoro -J eshiraishi@ponyo.tail9f8595.ts.net eshiraishi@totoro.mbzu.ae"

arm=$R/sft_full_3399_beam8.jsonl
if [ ! -f ${arm%.jsonl}.scored.jsonl ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=30 $TOTORO "cat ggrpo/runs/codenet_val_3399/beam8_gpu1.jsonl" > $arm.part
  ssh -o BatchMode=yes -o ConnectTimeout=30 eshiraishi@ponyo.tail9f8595.ts.net "cat gg/ggrpo/runs/codenet_val_3399/beam8_ponyo.jsonl" >> $arm.part
  mv $arm.part $arm
  echo "FETCHED $arm $(wc -l < $arm) rows $(date -u +%FT%TZ)" >> $S
  echo "SCORED $arm $(score $arm) $(date -u +%FT%TZ)" >> $S
fi

base=$R/gg_base_beam8_t517.jsonl
if [ ! -f ${base%.jsonl}.scored.jsonl ]; then
  cp $R/beam8_gpu1.jsonl $base.part
  ssh -o BatchMode=yes -o ConnectTimeout=30 eshiraishi@ponyo.tail9f8595.ts.net "cat gg/ggrpo/runs/codenet_val_base_t517/beam8_ponyo.jsonl" >> $base.part
  mv $base.part $base
  echo "FETCHED $base $(wc -l < $base) rows $(date -u +%FT%TZ)" >> $S
  echo "SCORED $base $(score $base) $(date -u +%FT%TZ)" >> $S
fi
echo "SCORE_DONE $(date -u +%FT%TZ)" >> $S
