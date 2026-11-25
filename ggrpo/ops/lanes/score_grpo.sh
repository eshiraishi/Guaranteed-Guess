#!/usr/bin/env bash
# Runs on the Mac: scores the GRPO run's two validation beam-8 evals -- step 184 first, then step 368, the headline
# number -- through the VM's reward server.
#
# Enzo, Nov 18 ("Wait for control"): the single reward VM is driving the permuted-reward control, a prereg arm of its
# own (H3), and scoring 383 problems through it would slow every remaining training step. So this lane waits for the
# control to exit before it scores anything, and scores each step as soon as both of its halves are complete.
#
# The gate is any RANDOM_EXIT, not RANDOM_EXIT 0: a failed control still frees the VM, and its exit code is recorded
# here rather than silently ignored. A non-zero code is a P0 -- the chained LR grid refuses to start without
# RANDOM_EXIT 0, so the Blackwell would be idle.
set -euo pipefail
cd "$(dirname "$0")/../.."
url=http://192.168.64.2:8765/score
R=results/codenet_val
T=$R/tasks_383.jsonl
S=$R/score_grpo.status
PONYO=eshiraishi@ponyo.tail9f8595.ts.net

# ponyo was the only way to totoro, and on 2026-11-19T19:39Z it went down for 12.5 h and took the reward tunnel, the
# training run and this lane with it. On campus totoro.mbzu.ae resolves, so go straight there and keep the jump only as
# the off-campus fallback -- the same two-path choice ops/lanes/tunnel.sh makes.
totoro() {
  if [ -n "$(dig +short +time=2 +tries=1 totoro.mbzu.ae 2>/dev/null)" ]; then
    echo "-o HostKeyAlias=totoro eshiraishi@totoro.mbzu.ae"
  else
    echo "-o HostKeyAlias=totoro -J eshiraishi@ponyo.tail9f8595.ts.net eshiraishi@totoro.mbzu.ae"
  fi
}

score() {  # score PREDICTIONS: row and id check, then ggrpo.score next to it
  uv run --frozen python -c "
import polars as pl
p, t = pl.read_ndjson('$1'), pl.read_ndjson('$T')
assert p.height == t.height and set(p['task_id']) == set(t['task_id']), (p.height, t.height)"
  uv run --frozen python -m ggrpo.score --server.url $url --score.tasks $T --score.predictions $1 \
    --score.out ${1%.jsonl}.scored.jsonl --score.reference $R/ref_383.scored.jsonl --log.file runs/ggrpo.jsonl 2>&1 | tail -n 1
}

wait_halves() {  # wait_halves REMOTE_DIR: both cards' halves through, or stop
  until ssh -o BatchMode=yes -o ConnectTimeout=30 $(totoro) "grep -qE '^(LANE_DONE|FAILED|SHORT|RECOVER_FAILED)' ggrpo/runs/$1/gpu1.status" \
     && ssh -o BatchMode=yes -o ConnectTimeout=30 $PONYO "grep -qE '^(LANE_DONE|FAILED|SHORT|RECOVER_FAILED)' gg/ggrpo/runs/$1/ponyo.status"; do sleep 300; done
  ssh -o BatchMode=yes -o ConnectTimeout=30 $(totoro) "grep -q '^LANE_DONE' ggrpo/runs/$1/gpu1.status" \
    || { echo "GPU1_FAILED $1 $(date -u +%FT%TZ)" >> $S; exit 1; }
  ssh -o BatchMode=yes -o ConnectTimeout=30 $PONYO "grep -q '^LANE_DONE' gg/ggrpo/runs/$1/ponyo.status" \
    || { echo "PONYO_FAILED $1 $(date -u +%FT%TZ)" >> $S; exit 1; }
}

fetch_and_score() {  # fetch_and_score REMOTE_DIR LOCAL_NAME
  local arm=$R/$2.jsonl
  [ -f ${arm%.jsonl}.scored.jsonl ] && { echo "SKIP $2 already scored $(date -u +%FT%TZ)" >> $S; return; }
  ssh -o BatchMode=yes -o ConnectTimeout=30 $(totoro) "cat ggrpo/runs/$1/beam8_gpu1.jsonl" > $arm.part
  ssh -o BatchMode=yes -o ConnectTimeout=30 $PONYO "cat gg/ggrpo/runs/$1/beam8_ponyo.jsonl" >> $arm.part
  mv $arm.part $arm
  echo "FETCHED $2 $(wc -l < $arm) rows $(date -u +%FT%TZ)" >> $S
  echo "SCORED $2 $(score $arm) $(date -u +%FT%TZ)" >> $S
}

echo "WAIT control $(date -u +%FT%TZ)" >> $S
until ssh -o BatchMode=yes -o ConnectTimeout=30 $(totoro) 'grep -q "^RANDOM_EXIT" ggrpo/runs/codenet_random_seed0.status'; do sleep 300; done
echo "CONTROL $(ssh -o BatchMode=yes -o ConnectTimeout=30 $(totoro) 'grep "^RANDOM_EXIT" ggrpo/runs/codenet_random_seed0.status' | tail -1) $(date -u +%FT%TZ)" >> $S

wait_halves codenet_val_grpo184
fetch_and_score codenet_val_grpo184 grpo184_beam8

wait_halves codenet_val_grpo368
fetch_and_score codenet_val_grpo368 grpo368_beam8
echo "SCORE_DONE $(date -u +%FT%TZ)" >> $S
