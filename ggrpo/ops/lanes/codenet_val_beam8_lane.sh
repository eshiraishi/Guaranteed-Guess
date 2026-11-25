#!/usr/bin/env bash
# Beam 8, CodeNet validation, on any finished model. The same lane as codenet_val_11328_lane.sh, with the model and the
# output folder as arguments instead of hardcoded (Enzo, Nov 18: evaluate the GRPO step-184 checkpoint), so one script
# serves every checkpoint we score on this benchmark.
# One shard per 4070, the same halves each card ran for the GG base and for the full-data SFT: gpu1 192 tasks, ponyo
# 191. Generation only; scored from the Mac afterwards, so this lane never touches the reward VM that training is using.
# Usage: lane.sh <name> <repo dir> <python> <model> <out dir>. Env comes from the caller.
# A task that runs out of memory goes to oom_<name>.txt and leaves the shard, then the lane goes on. Any other failure
# stops it. When the shard is through, every dropped task is re-generated with the KV cache in host RAM
# (cache_implementation offloaded: identical text, 6.1x the time), so the half ends at its full row count and nothing
# is excluded. The lane asserts that count before reporting LANE_DONE.
set -uo pipefail
name=$1; cd "$2"; py=$3; model=$4; O=$5
tasks=$O/val_$name.jsonl; pristine=$O/pristine_$name.jsonl; out=$O/beam8_$name.jsonl
S=$O/$name.status; L=$O/$name.log; A=$O/$name.attempt
echo "LANE_START $model $(date -u +%FT%TZ)" >> $S
until $py -m ggrpo.generate --generate.model $model --data.eval $tasks --generate.out $out > $A 2>&1; do
  code=$?; cat $A >> $L
  if ! grep -q "OutOfMemoryError\|out of memory" $A; then echo "FAILED $code $(date -u +%FT%TZ)" >> $S; exit 1; fi
  $py - $tasks $out $O/oom_$name.txt <<'PY'
import json, os, sys
tasks, out, oom = sys.argv[1:]
done = {json.loads(l)["task_id"] for l in open(out)} if os.path.exists(out) else set()
lines = open(tasks).read().splitlines()
index = next(i for i, l in enumerate(lines) if json.loads(l)["task_id"] not in done)
open(oom, "a").write(json.loads(lines[index])["task_id"] + "\n")
open(tasks, "w").write("".join(l + "\n" for i, l in enumerate(lines) if i != index))
PY
  echo "OOM $(tail -1 $O/oom_$name.txt) $(date -u +%FT%TZ)" >> $S
done
cat $A >> $L
if [ -s $O/oom_$name.txt ]; then
  echo "RECOVER_START $(wc -l < $O/oom_$name.txt) $(date -u +%FT%TZ)" >> $S
  $py - $pristine $O/oom_$name.txt $O/val_oom_$name.jsonl <<'PY'
import json, sys
pristine, oom, out = sys.argv[1:]
wanted = set(open(oom).read().split())
rows = [l for l in open(pristine) if json.loads(l)["task_id"] in wanted]
assert len(rows) == len(wanted), (len(rows), len(wanted))
open(out, "w").write("".join(rows))
PY
  $py -m ggrpo.generate --generate.model $model --generate.beam_search.cache_implementation offloaded \
    --data.eval $O/val_oom_$name.jsonl --generate.out $O/beam8_oom_$name.jsonl >> $L 2>&1 \
    || { echo "RECOVER_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
  cat $O/beam8_oom_$name.jsonl >> $out
  echo "RECOVER_DONE $(date -u +%FT%TZ)" >> $S
fi
rows=$(wc -l < $out); want=$(wc -l < $pristine)
[ "$rows" -eq "$want" ] || { echo "SHORT $rows/$want $(date -u +%FT%TZ)" >> $S; exit 1; }
echo "LANE_DONE $rows rows $(date -u +%FT%TZ)" >> $S
