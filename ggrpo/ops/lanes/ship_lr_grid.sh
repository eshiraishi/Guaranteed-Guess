#!/usr/bin/env bash
# Runs on the Mac: once the imitation arm's LR grid finishes all three runs, streams each arm's final model from totoro
# to ponyo over the LAN and starts ponyo's half of the arms' beam-8 eval. Neither GPU host can ssh the other on its
# own, so the Mac forwards its agent for the length of each transfer (ops/lanes/ship_11328.sh, Nov 17).
set -uo pipefail
cd "$(dirname "$0")/../.."
S=results/codenet_val/ship_lr_grid.status
TOTORO="-o HostKeyAlias=totoro -J eshiraishi@ponyo.tail9f8595.ts.net eshiraishi@totoro.mbzu.ae"
PONYO=eshiraishi@ponyo.tail9f8595.ts.net
echo "WAIT grid $(date -u +%FT%TZ)" >> $S
until ssh -o BatchMode=yes -o ConnectTimeout=30 $TOTORO 'grep -qE "^(GRID_EXIT|GRID_ABORT)" ggrpo/runs/sft_pool_references.status'; do sleep 300; done
ssh -o BatchMode=yes -o ConnectTimeout=30 $TOTORO 'grep -q "^GRID_EXIT 0" ggrpo/runs/sft_pool_references.status' \
  || { echo "GRID_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
for lr in 5e-6 1e-5 2e-5; do
  arm=runs/sft_pool_refs_lr$lr
  for attempt in 1 2 3 4 5; do
    ssh -A -o ServerAliveInterval=15 $PONYO "mkdir -p ~/gg/ggrpo/$arm && rsync -a --partial --exclude 'checkpoint-*' \
      -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' eshiraishi@totoro.mbzu.ae:ggrpo/$arm/ ~/gg/ggrpo/$arm/" && break
    echo "COPY_RETRY $lr $attempt $(date -u +%FT%TZ)" >> $S
    sleep 60
  done
  a=$(ssh -o BatchMode=yes -o ConnectTimeout=30 $TOTORO "sha256sum ggrpo/$arm/model.safetensors" | cut -d" " -f1)
  b=$(ssh -o BatchMode=yes -o ConnectTimeout=30 $PONYO "sha256sum gg/ggrpo/$arm/model.safetensors" | cut -d" " -f1)
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "HASH_MISMATCH lr=$lr $a $b $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "COPIED lr=$lr $a $(date -u +%FT%TZ)" >> $S
done
ssh -o BatchMode=yes -o ConnectTimeout=30 $PONYO 'tmux new-session -d -s e1grid "bash ~/gg/ggrpo/runs/e1_lr_grid_ponyo.sh"'
echo "PONYO_STARTED $(date -u +%FT%TZ)" >> $S
