#!/usr/bin/env bash
# Runs on the Mac (nohup) beside tunnel.sh. When the Mac drops off the network (DNS outages, Oct 23), totoro's end of
# the old reverse tunnel can outlive it and keep 127.0.0.1:8765 bound: every reconnect then fails with "remote port
# forwarding failed" and training waits on rewards (13:20-13:33 GST that day). Once the last 3 attempts all failed that
# way and totoro's 8765 does not answer, this kills the stale sshd, but only if it is the single tunnel session
# ("sshd: USER", no tty or command); anything else is logged for a person to look at.
here=$(cd "$(dirname "$0")/../.." && pwd)
while sleep 60; do
  t=$(grep -E 'tunnel (start|exit)|forwarding failed' "$here/runs/tunnel.log" | tail -n 9)
  [ "$(echo "$t" | grep -c 'forwarding failed')" = 3 ] && [[ $(echo "$t" | tail -n 1) == "tunnel exit"* ]] || continue
  ssh -o BatchMode=yes -o ConnectTimeout=20 totoro.mbzu.ae 'curl -s -o /dev/null -m 10 http://127.0.0.1:8765/ && exit
    p=$(pgrep -u $USER -xf "sshd: $USER")
    [ "$(echo "$p" | grep -c .)" = 1 ] && kill $p && echo "KILLED stale tunnel sshd $p" || echo "NOT_ONE tunnel sshd: $p"' |
    sed "s/^/$(date -u +%FT%TZ) /"
done
