#!/usr/bin/env bash
# Runs on the Mac: prints a one-line-per-lane digest of every job; with --until-change, polls every 3 min and exits
# (printing old and new digest) as soon as anything changes. Unreachable hosts keep their previous line.
here=$(cd "$(dirname "$0")/../.." && pwd)
digest() {
  local t p c
  t=$(ssh -o BatchMode=yes -o ConnectTimeout=20 totoro.mbzu.ae 'cd ~/ggrpo/runs
    echo "train: $(tail -n 1 full.status) | tracebacks $(grep -c Traceback full.log) | ckpts $(ls -d full/checkpoint-* 2>/dev/null | sed "s/.*-//" | sort -n | tr "\n" " ")"
    echo "seed1: $(tail -n 1 full_seed1.status 2>/dev/null) | tracebacks $(grep -c Traceback full_seed1.log 2>/dev/null) | ckpts $(ls -d full_seed1/checkpoint-* 2>/dev/null | sed "s/.*-//" | sort -n | tr "\n" " ")"
    echo "gpu1: $(grep -q DEV_BASE_DONE modern_dev_beam_search.log && echo base-dev-done || echo base-dev-running) | $(tail -n 1 eval_gpu1.status)"
    echo "gpu1-seed1: $(tail -n 1 eval_gpu1_seed1.status 2>/dev/null)"
    echo "chain: $(tail -n 1 chain.status 2>/dev/null)"
    for n in dsm_seed0 drgrpo_seed0; do
      echo "$n: $(tail -n 1 $n.status 2>/dev/null) | tracebacks $(grep -c Traceback $n.log 2>/dev/null) | step $(tr "\r" "\n" 2>/dev/null < $n.log | grep -oE "[0-9]+/106" | tail -n 1) | ckpts $(ls -d $n/checkpoint-* 2>/dev/null | sed "s/.*-//" | sort -n | tr "\n" " ")| gpu1 $(tail -n 1 eval_gpu1_$n.status 2>/dev/null)"
    done') || t=$prev_t
  p=$(ssh -o BatchMode=yes -o ConnectTimeout=20 ponyo 'cd ~/gg/ggrpo/runs
    echo "ponyo: $(grep -q SEED1_DONE paper_seed1.log && echo seed1-done || echo seed1-running) | relayed $(ls full/checkpoint-*/.relayed 2>/dev/null | grep -o "[0-9]*/" | tr -d / | tr "\n" " ")| $(tail -n 1 eval_ponyo.status)"
    echo "ponyo-seed1: relayed $(ls full_seed1/checkpoint-*/.relayed 2>/dev/null | grep -o "[0-9]*/" | tr -d / | tr "\n" " ")| $(tail -n 1 eval_ponyo_seed1.status 2>/dev/null)"
    for n in dsm_seed0 drgrpo_seed0; do echo "ponyo-$n: relayed $(ls $n/checkpoint-*/.relayed 2>/dev/null | grep -o "[0-9]*/" | tr -d / | tr "\n" " ")| $(tail -n 1 eval_ponyo_$n.status 2>/dev/null)"; done') || p=$prev_p
  c=$(ssh -o BatchMode=yes -o ConnectTimeout=20 student-lab 'echo "cluster: $(squeue -h -u enzo.shiraishi -o "%j=%T" | sort | tr "\n" " ")"') || c=$prev_c
  prev_t=$t prev_p=$p prev_c=$c
  printf '%s\n%s\n%s\n' "$t" "$p" "$c"
  echo "mac: tunnel exits $(grep -c 'tunnel exit' "$here/runs/tunnel.log" 2>/dev/null) | relay $(tail -n 1 "$here/runs/relay.log" 2>/dev/null) | cluster relay $(grep -E 'PUSHED|VIEW_OK|SUBMIT' "$here/runs/relay_cluster.log" 2>/dev/null | tail -n 1) | seed1 relay $(tail -n 1 "$here/runs/relay_seed1.log" 2>/dev/null) | dsm relay $(tail -n 1 "$here/runs/relay_dsm_seed0.log" 2>/dev/null) | drgrpo relay $(tail -n 1 "$here/runs/relay_drgrpo_seed0.log" 2>/dev/null) | dsm score $(grep -cE '^SCORED|^COPIED' "$here/runs/score_dsm_seed0.log" 2>/dev/null)/10 $(grep -c SCORE_FAILED "$here/runs/score_dsm_seed0.log" 2>/dev/null) failed | battery $(pmset -g batt | grep -o '[0-9]*%;[^;]*' | head -1 | awk -F'[%;]' '{print ($1 < 20 && $3 ~ /discharging/) ? "LOW" : "ok"}')"
}
old=$(digest)
[ "$1" = --until-change ] || { echo "$old"; exit; }
while sleep 180; do
  new=$(digest)
  [ "$new" != "$old" ] && { echo "$(date -u +%FT%TZ) changed:"; diff <(echo "$old") <(echo "$new") | grep '^[<>]'; exit; }
done
