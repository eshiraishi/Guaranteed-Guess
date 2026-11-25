#!/usr/bin/env bash
# Runs on the Mac beside tunnel.sh and tunnel_guard.sh: restarts the gg-ventura VM whenever it stops.
#
# The reward daemon inside the VM (ops/vm/ae.mbzuai.ggrpo.reward.plist) brings the server back after a VM boot or a
# server crash, but it cannot help when the VM itself is gone -- which is the failure that has actually cost us time:
# Nov 14 a Mac sleep on battery stopped it for 6.6 h, Nov 17 it was found stopped again after 2.5 h of the census and
# the 11328 eval retrying /score once a minute. tart does not restart a stopped VM, and nothing else watches it.
#
# Starting the VM is enough on its own: the daemon starts the server at boot and tunnel.sh reconnects within 10 s.
# caffeinate -ims holds off idle sleep for as long as the VM runs; it does not stop a closed-lid or manual sleep, so
# the Mac still has to stay on AC. Every start and every state change is logged for a person to read afterwards.
set -uo pipefail
tart=$HOME/tools/tart/tart.app/Contents/MacOS/tart
here=$(cd "$(dirname "$0")/../.." && pwd)
log=$here/runs/vm_guard.log
stamp() { date -u +%FT%TZ; }

while sleep 60; do
  state=$("$tart" list 2>/dev/null | awk '$2=="gg-ventura"{print $NF}')
  if [ -z "$state" ]; then
    echo "$(stamp) NO_VM gg-ventura missing from tart list" >> "$log"
    continue
  fi
  [ "$state" = "running" ] && continue
  # a start already in flight reports a non-running state for ~20 s; do not launch a second tart
  pgrep -f "tart run --no-graphics gg-ventura" > /dev/null && continue
  echo "$(stamp) VM_DOWN state=$state, starting" >> "$log"
  nohup caffeinate -ims "$tart" run --no-graphics gg-ventura >> "$here/runs/tart.log" 2>&1 &
  for _ in $(seq 1 12); do
    sleep 10
    ip=$("$tart" ip gg-ventura 2>/dev/null) && [ -n "$ip" ] && break
  done
  if [ -n "${ip:-}" ]; then
    echo "$(stamp) VM_UP $ip" >> "$log"
  else
    echo "$(stamp) VM_START_FAILED after 120 s -- needs a person" >> "$log"
  fi
done
