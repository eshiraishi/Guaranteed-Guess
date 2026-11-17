#!/usr/bin/env bash
# Runs on the Mac beside tunnel.sh, tunnel_guard.sh and vm_guard.sh: samples every lane every 5 minutes so the board
# has a history to measure against (Enzo, Nov 17).
#
# ggrpo.board derives its rate and its STALLED verdict from runs/board_history.tsv, so a board run with no prior
# samples can only print progress -- no rate, no ETA. This keeps the history filling whether or not anyone is looking,
# which also means the record survives to say afterwards how long a lane actually took and where it stalled.
# The output is discarded: the point is the appended samples, and `python -m ggrpo.board` prints the table on demand.
cd "$(cd "$(dirname "$0")/../.." && pwd)"
while true; do
  .venv/bin/python -m ggrpo.board > /dev/null 2>> runs/board_poll.log
  sleep 300
done
