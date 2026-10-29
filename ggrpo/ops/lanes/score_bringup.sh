#!/usr/bin/env bash
# Runs on the Mac: scores one run's cluster BringUp generations (relay_arm.sh / relay_run.sh) as for seed 0. Pulls the
# a/b shards from student-lab into results/bringup/cluster/NAME_STEP/, joins them (66 programs, checked by
# bringup_to_preds.py), and runs the upstream harness (bringup_score.py) in the VM at the lowest CPU priority, since
# the VM also serves training rewards. Non-interactive ssh lacks Homebrew's `timeout`, which the Makefile needs (without
# it every program fails), hence the PATH. Writes OUT/checkpoint-STEP_bringup_MODE.scored.jsonl.
# usage: score_bringup.sh NAME STEP OUT MODE...   e.g. score_bringup.sh sft_seed0 108 results/grpo_eval/sft-seed0 beam8
cd "$(dirname "$0")/../.."
name=$1 n=$2 out=$3; shift 3
d=results/bringup/cluster/${name}_$n tag=${name}_${n}_bringup
mkdir -p $d $out
for m in "$@"; do
  scp -q student-lab:cs8001-bringup/$name/${m}_a.jsonl student-lab:cs8001-bringup/$name/${m}_b.jsonl $d/ &&
    cat $d/${m}_a.jsonl $d/${m}_b.jsonl > $d/$m.jsonl &&
    python3 ops/mac-scratch/643399c8/bringup_to_preds.py $d/$m.jsonl $d/${tag}_$m.preds.json &&
    scp -q $d/${tag}_$m.preds.json admin@192.168.64.2:scratch/ &&
    ssh -o BatchMode=yes admin@192.168.64.2 "cd ~/scratch && PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH nice -n 20 \
      python3 bringup_score.py ${tag}_$m.preds.json ${tag}_$m.scored.jsonl > ${tag}_$m.score.log 2>&1" &&
    scp -q admin@192.168.64.2:scratch/${tag}_$m.scored.jsonl $out/checkpoint-${n}_bringup_$m.scored.jsonl &&
    echo "SCORED $out/checkpoint-${n}_bringup_$m.scored.jsonl pass $(grep -c '"verdict": "pass"' $out/checkpoint-${n}_bringup_$m.scored.jsonl)/$(wc -l < $out/checkpoint-${n}_bringup_$m.scored.jsonl) $(date -u +%FT%TZ)" ||
    echo "SCORE_FAILED $name $n $m $(date -u +%FT%TZ)"
done
