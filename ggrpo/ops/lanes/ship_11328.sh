#!/usr/bin/env bash
# Runs on the Mac: once the full-data SFT exits cleanly, streams its final model from totoro to ponyo (tar through this
# host, no local copy; save_model writes no optimizer state, so this is ~6.2 GB, not the 18 GB of a checkpoint folder),
# then starts ponyo's half of the start check. Neither GPU host can ssh the other, so the Mac is the relay.
set -euo pipefail
cd "$(dirname "$0")/../.."
TOTORO="-o HostKeyAlias=totoro -J eshiraishi@ponyo.tail9f8595.ts.net eshiraishi@totoro.mbzu.ae"
PONYO=eshiraishi@ponyo.tail9f8595.ts.net
S=results/codenet_val/ship_11328.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until ssh $TOTORO 'grep -q "^SFT_EXIT" ~/ggrpo/runs/sft_full_seed0.status'; do sleep 120; done
ssh $TOTORO 'grep -q "^SFT_EXIT 0" ~/ggrpo/runs/sft_full_seed0.status' \
  || { echo "SFT_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
until ssh $TOTORO '[ -s ~/ggrpo/runs/sft_full_seed0/model.safetensors ]'; do sleep 30; done
sleep 60   # save_model writes the weights before the tokenizer files; do not read the folder mid-write
echo "COPY_START $(date -u +%FT%TZ)" >> $S
# Nov 17: ponyo pulls from totoro over the LAN instead of the Mac relaying a tar pipe between them.
# The pipe failed three times mid-copy (ponyo off the tailnet, then a Mac disconnection), and `set -e` on a broken
# pipe killed the whole lane each time, so the eval half never started. It was also slow: ~0.9 MB/s through the Mac,
# against 85 MB/s host to host, because ponyo reaches totoro directly at 10.127.102.200.
# -A forwards the Mac's agent, so ponyo authenticates to totoro for the length of the transfer and neither host gets
# a new authorized_keys entry or a changed known_hosts (ponyo already trusts totoro's key, same fingerprint the Mac
# has). rsync resumes a partial file and retries, so a dropped link costs seconds rather than the lane.
for attempt in 1 2 3 4 5; do
  ssh -A $PONYO 'mkdir -p ~/gg/ggrpo/runs/sft_full_seed0 && rsync -a --partial --exclude "checkpoint-*" \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    eshiraishi@totoro.mbzu.ae:ggrpo/runs/sft_full_seed0/ ~/gg/ggrpo/runs/sft_full_seed0/' && break
  echo "COPY_RETRY $attempt $(date -u +%FT%TZ)" >> $S
  sleep 60
done
ssh $PONYO 'ls -l ~/gg/ggrpo/runs/sft_full_seed0/model.safetensors'
# the halves must read the same bytes: a truncated stream would silently change the arm
a=$(ssh $TOTORO 'sha256sum ~/ggrpo/runs/sft_full_seed0/model.safetensors' | cut -d" " -f1)
b=$(ssh $PONYO 'sha256sum ~/gg/ggrpo/runs/sft_full_seed0/model.safetensors' | cut -d" " -f1)
[ "$a" = "$b" ] || { echo "HASH_MISMATCH $a $b $(date -u +%FT%TZ)" >> $S; exit 1; }
echo "COPY_DONE $a $(date -u +%FT%TZ)" >> $S
ssh $PONYO 'tmux new-session -d -s beam11328 "bash ~/gg/ggrpo/runs/beam8_11328_ponyo.sh"'
echo "PONYO_STARTED $(date -u +%FT%TZ)" >> $S
