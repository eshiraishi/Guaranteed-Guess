#!/usr/bin/env bash
# Runs on the Mac (nohup, with caffeinate -ims): reverse tunnel totoro:8765 -> VM reward server, reconnecting 10 s after
# any exit. Training's reward.py retries for up to 12 h, so a drop only pauses a step. Full hostname: the short alias
# does not resolve off the campus network.
while true; do
  echo "tunnel start $(date -u +%T)"
  ssh -N -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -R 8765:192.168.64.2:8765 totoro.mbzu.ae
  echo "tunnel exit $? $(date -u +%T)"; sleep 10
done
