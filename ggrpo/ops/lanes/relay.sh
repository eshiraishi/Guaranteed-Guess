#!/usr/bin/env bash
# Runs on the Mac (nohup): copies eval-relevant checkpoint files totoro -> ponyo for the ponyo lane.
# ponyo pulls directly from totoro over the campus network (~25 MB/s) with this Mac's forwarded ssh agent;
# the Mac's own link (~2.7 MB/s, shared with the reward tunnel) never carries the weights.
# totoro refuses ssh from the student-lab cluster, so the cluster gets checkpoint-106 from ponyo (relay_cluster.sh).
keep="--include=*.json --include=*.jinja --include=model.safetensors --exclude=*"   # no optimizer/scheduler/rng state
for n in 81 106; do
  src=ggrpo/runs/full/checkpoint-$n dst=gg/ggrpo/runs/full/checkpoint-$n
  until ssh -o BatchMode=yes -o ConnectTimeout=20 totoro.mbzu.ae "test -f $src/trainer_state.json"; do sleep 120; done
  sleep 180   # let the save finish
  until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo "mkdir -p $dst && rsync -a $keep totoro.mbzu.ae:$src/ $dst/ && touch $dst/.relayed"; do
    echo "relay $n failed $(date -u +%FT%TZ), retrying"; sleep 120
  done
  echo "RELAYED $n $(date -u +%FT%TZ)"
done
