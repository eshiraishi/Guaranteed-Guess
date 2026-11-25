#!/usr/bin/env bash
# Mac (nohup caffeinate -ims): scores every finished val E2 file from totoro (Nov 23). The GPU hosts only generate
# (ggrpo.sample --sample.score false), so no eval waits on the reward tunnel; this loop pulls the files, scores the
# unscored ones through the VM (ggrpo.score_samples) and adds the official pass (ggrpo.official, AOJ val tasks).
# Replaces e2_recheck_mac.sh: the differential suite was dropped on Nov 22. A failure stops the loop.
# Usage: e2_score_mac.sh RESULTS_DIR TASKS REMOTE_DIR... (Nov 25: one loop per task set), e.g.
#   e2_score_mac.sh results/codenet_val 383 e2_phase1 e2_phase2
#   e2_score_mac.sh results/codeforces_val 359 e2_codeforces
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$1 TASKS=$2; shift 2
D=$R/e2
S=$D/score.status
mkdir -p $D/raw $R/official
while true; do
  rsync -a -e "ssh -o HostKeyAlias=totoro -o BatchMode=yes" \
    $(for dir in "$@"; do echo "eshiraishi@totoro.mbzu.ae:ggrpo/runs/$dir/*_sample16.jsonl"; done) \
    $D/raw/ 2>> $S
  for f in $D/raw/*_sample16.jsonl; do
    name=$(basename $f .jsonl); out=$D/$name.jsonl
    if [ ! -s $out ]; then
      # complete = all val tasks, each with the same number of samples
      uv run python -c "
import sys, polars as pl
n = pl.read_ndjson(sys.argv[1], infer_schema_length=None).group_by('task_id').len()['len']
sys.exit(0 if n.len() == int(sys.argv[2]) and n.n_unique() == 1 else 1)" $f $TASKS || continue
      if head -1 $f | grep -q '"passed"'; then cp $f $out; echo "COPIED_SCORED $name $(date -u +%FT%TZ)" >> $S
      else
        echo "SCORE_START $name $(date -u +%FT%TZ)" >> $S
        uv run python -m ggrpo.score_samples --score_samples.samples $f --score_samples.out $out >> $S 2>&1 \
          || { echo "SCORE_FAILED $name $(date -u +%FT%TZ)" >> $S; exit 1; }
      fi
    fi
    [ "$(wc -l < $out)" -eq "$(wc -l < $f)" ] || { echo "ROWS_MISMATCH $name $(date -u +%FT%TZ)" >> $S; exit 1; }
    official=$R/official/${name}_e2.jsonl
    [ -s $official ] && continue
    uv run python -m ggrpo.official --official.scored $out --official.column raw --official.out $official >> $S 2>&1 \
      || { echo "OFFICIAL_FAILED $name $(date -u +%FT%TZ)" >> $S; exit 1; }
  done
  sleep 600
done
