#!/usr/bin/env bash
# Mac (nohup caffeinate -ims): the registered pass for every val E2 (Enzo, Nov 21: select on the registered pass, not
# sample tests alone). Every 30 min, copies the complete E2 files (6,128 rows) from totoro and re-runs each new one's
# sample-test passes against the differential suite with the x86 voter (ggrpo.recheck), next to the E2 file.
# Validation only. A failed recheck stops the loop.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/codenet_val/e2
S=$D/recheck.status
mkdir -p $D
while true; do
  rsync -a -e "ssh -o HostKeyAlias=totoro -o BatchMode=yes" \
    'eshiraishi@totoro.mbzu.ae:ggrpo/runs/e2_phase1/*_sample16.jsonl' 'eshiraishi@totoro.mbzu.ae:ggrpo/runs/e2_phase2/*_sample16.jsonl' \
    'eshiraishi@totoro.mbzu.ae:ggrpo/runs/e2_pool_refs_lr*_sample16.jsonl' $D/ 2>> $S
  for f in $D/*_sample16.jsonl; do
    out=${f%.jsonl}.recheck_x86.jsonl
    [ -s $out ] && continue
    [ "$(wc -l < $f)" -eq 6128 ] || continue
    echo "RECHECK_START $f $(date -u +%FT%TZ)" >> $S
    uv run python -m ggrpo.recheck --recheck.scored $f --recheck.column raw \
      --recheck.suites data/codenet/differential_val_x86.jsonl --recheck.out $out --recheck.workers 6 >> $S 2>&1 \
      || { echo "RECHECK_FAILED $f $(date -u +%FT%TZ)" >> $S; rm -f $out; exit 1; }
  done
  sleep 1800
done
