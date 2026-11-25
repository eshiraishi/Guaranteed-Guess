#!/usr/bin/env bash
# Runs on the Mac (nohup, with caffeinate -ims): reverse tunnel totoro:8765 -> VM reward server, reconnecting 10 s after
# any exit. Training's reward.py retries for up to 12 h, so a drop only pauses a step. totoro is not on Tailscale and
# totoro.mbzu.ae resolves only on the campus network, so off campus the only way in is the jump through ponyo (Nov 15,
# the Mac off campus: 'Could not resolve hostname totoro.mbzu.ae' every 10 s, totoro's 8765 unbound).
#
# ponyo is the single point of failure in that path: it went offline at 2026-11-19T19:39:23Z and training stalled with
# nothing on this side able to fix it. So when the campus name resolves, go straight to totoro and leave ponyo out.
while true; do
  if [ -n "$(dig +short +time=2 +tries=1 totoro.mbzu.ae 2>/dev/null)" ]; then
    echo "tunnel start direct $(date -u +%T)"
    ssh -N -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -o HostKeyAlias=totoro \
      -R 8765:192.168.64.2:8765 eshiraishi@totoro.mbzu.ae
  else
    echo "tunnel start via ponyo $(date -u +%T)"
    ssh -N -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -o HostKeyAlias=totoro -J eshiraishi@ponyo.tail9f8595.ts.net \
      -R 8765:192.168.64.2:8765 eshiraishi@totoro.mbzu.ae
  fi
  echo "tunnel exit $? $(date -u +%T)"; sleep 10
done
