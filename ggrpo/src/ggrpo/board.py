"""The state of every lane, in one table, run from the Mac.

Enzo, Nov 17: "so i dont have to keep asking you". Every lane already writes an append-only runs/<name>.status with
ISO-8601 UTC lines and grows a .jsonl row by row, on whichever host it runs, so progress, elapsed, rate and ETA are
derivable without touching a single lane script. The lanes, hosts and thresholds are [board] in config.toml.

Run it from the Mac, because no other host sees everything: totoro has GPU0's census and GPU1's eval but not ponyo's
half, and only the Mac sees the reward VM, the tunnel and the guards.

The rate comes from the last board.rate_window_minutes of samples this writes to board.history, never from total
elapsed: elapsed would have read the census at 3.9 tasks/min on Nov 17 against a true 8.0, because the 2.5 h
reward-tunnel stall that morning is folded into it. The same history makes a stall visible -- rows frozen past
board.stall_after_minutes read STALLED and for how long, the failure that cost 6.6 h on Nov 14 and 2.5 h on Nov 17,
and that a progress number hides, since vLLM holds the GPU at 100% generating completions it cannot score.
"""

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ggrpo.settings import Board, BoardLane, Settings

# a lane whose last status line starts with one of these has stopped; anything else is still working
DONE_PREFIXES = ("LANE_DONE", "SHARD_EXIT", "SAMPLED_DONE", "PROBE_EXIT", "EXIT", "DONE", "FAILED",
                 "STOPPED",      # STOPPED is written by hand when a lane is killed, so the board stops calling it
                                 # running; a restart appends START after it
                 "GRPO_EXIT", "PASS2_EXIT", "RANDOM_EXIT", "GRID_EXIT", "GRID_ABORT", "SFT_EXIT", "FULL_EXIT",
                 "SHORT", "HASH_MISMATCH", "KILLED", "REFUSED")
# Nov 18: the training lanes spell their own exit marker, so "EXIT" as a prefix never matched GRPO_EXIT 0 and the
# finished run read STALLED. Every "<step>_FAILED" marker is terminal too, and there are a dozen of them; ARM_EXIT is
# deliberately absent, since the LR grid writes one per arm and goes on to the next.
# A marker ending in DONE is not enough on its own: RECOVER_DONE, GEN_DONE, SCORE_DONE and COPY_DONE are all mid-lane.


def has_stopped(last: str) -> bool:
    return last.startswith(DONE_PREFIXES) or last.split(" ")[0].endswith("_FAILED")


def build_remote_script(board: Board, host: str) -> str:
    """One sh command per host, so a poll costs one round trip rather than one per lane."""
    parts = []
    for lane in board.lanes:
        if lane.host != host:
            continue
        repo = board.hosts[host].repo
        files = " ".join(str(repo / f) for f in (lane.rows, lane.rows_extra) if f)
        count = f"cat {files} 2>/dev/null | sort -u | wc -l | tr -d ' '" if lane.rows else "echo -1"
        parts.append(f'printf "%s\\t%s\\t%s\\n" "{lane.name}" "$({count} || echo -1)" '
                     f'"$(tail -1 {repo / lane.status} 2>/dev/null)"')
    parts.append('printf "GPU\\t%s\\t-\\n" '
                 '"$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | paste -sd" " -)"')
    return "; ".join(parts)


def poll_host(board: Board, host: str) -> tuple[dict[str, tuple[int, str]], str, str]:
    """Returns {lane: (rows, last status line)}, the GPU line, and an error that is empty on success."""
    done = subprocess.run(board.hosts[host].ssh + [build_remote_script(board, host)],
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        said = done.stderr.strip().splitlines()
        return {}, "", said[-1] if said else f"ssh exit {done.returncode}"
    rows, gpu = {}, ""
    for line in done.stdout.splitlines():
        name, count, status = (line.split("\t") + ["", ""])[:3]
        if name == "GPU":
            gpu = count.strip()
        else:
            rows[name] = (int(count or -1), status)
    return rows, gpu, ""


def read_history(path: Path) -> list[tuple[datetime, str, int]]:
    if not path.exists():
        return []
    samples = []
    for line in path.read_text().splitlines():
        stamp, lane, count = line.split("\t")
        samples.append((datetime.fromisoformat(stamp), lane, int(count)))
    return samples


def append_history(path: Path, now: datetime, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for lane, count in sorted(counts.items()):
            handle.write(f"{now.isoformat()}\t{lane}\t{count}\n")


def measure_rate(history, lane: str, now: datetime, count: int, window: timedelta) -> float | None:
    """Rows per minute over the window. None until two samples sit at least a minute apart."""
    seen = [(when, rows) for when, name, rows in history if name == lane and now - when <= window]
    if not seen:
        return None
    oldest_when, oldest_count = min(seen)
    minutes = (now - oldest_when).total_seconds() / 60
    return (count - oldest_count) / minutes if minutes >= 1 else None


def read_status_time(last: str) -> datetime | None:
    """The ISO-8601 UTC stamp a lane status line ends with, which every lane script writes."""
    for word in reversed(last.split()):
        try:
            return datetime.fromisoformat(word.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def measure_stall(history, lane: str, now: datetime, count: int, after: timedelta,
                  since: datetime | None) -> timedelta | None:
    """How long the row count has been frozen, once that passes the threshold.

    Frozen time is counted from the lane's own last status line, never from before it: a lane stopped and restarted
    has an unchanged row count across the gap, and counting that gap called a freshly restarted 9064 eval STALLED for
    2 h 14 (Nov 17). The restart's START line is the honest zero.
    """
    held = [when for when, name, rows in history if name == lane and rows == count]
    if not held:
        return None
    started = max([min(held)] + ([since] if since else []))
    waited = now - started
    return waited if waited >= after else None


def describe_local(board: Board) -> list[str]:
    """The Mac's own half: the reward VM and the helpers no remote host can see."""
    tart = Path(board.local.tart).expanduser()
    listed = subprocess.run([str(tart), "list"], capture_output=True, text=True)
    state = next((line.split()[-1] for line in listed.stdout.splitlines()
                  if len(line.split()) > 1 and line.split()[1] == board.local.vm), "MISSING")
    lines = [f"{'reward VM':<17}{state}"]
    for helper in board.local.helpers:
        alive = subprocess.run(["pgrep", "-f", f"ops/lanes/{helper}"], capture_output=True).returncode == 0
        lines.append(f"{helper:<17}{'running' if alive else 'DOWN'}")
    return lines


def describe_lane(board: Board, lane: BoardLane, polled: dict, history, now: datetime) -> str:
    rows, _gpu, error = polled[lane.host]
    if error:
        return f"{lane.name:<17}{'?':<10}{lane.host} unreachable: {error}"
    count, last = rows.get(lane.name, (-1, ""))
    if not last:
        state = "queued"        # no status file yet: a lane still behind another in its chain
    elif has_stopped(last):
        state = "done"
    elif last.startswith("WAIT"):
        state = "waiting"
    else:
        state = "running"
    if count < 0:
        return f"{lane.name:<17}{state:<10}{last or 'no output yet'}"
    recovering = last.startswith("RECOVER_START")
    minutes = (lane.recovering_stall_after_minutes if recovering and lane.recovering_stall_after_minutes
               else lane.stall_after_minutes or board.stall_after_minutes)
    after = timedelta(minutes=minutes)
    stalled = (measure_stall(history, lane.name, now, count, after, read_status_time(last))
               if state == "running" else None)
    if stalled:
        state = "STALLED"
    window = timedelta(minutes=lane.rate_window_minutes or board.rate_window_minutes)
    rate = measure_rate(history, lane.name, now, count, window)
    progress = f"{count // lane.rows_per_unit} / {lane.total // lane.rows_per_unit} {lane.unit}"
    if lane.total:
        progress += f" {100 * count / lane.total:.0f}%"
    if rate and rate > 0:
        shown = f"{rate / lane.rows_per_unit:.1f}/min"
    elif stalled:
        shown = f"frozen {str(stalled).split('.')[0]}"
    else:
        shown = "-"
    eta = ""
    if state == "running" and rate and rate > 0 and count < lane.total:
        when = now + timedelta(minutes=(lane.total - count) / rate)
        local = when.astimezone(timezone(timedelta(hours=board.report_utc_offset_hours)))
        eta = f"{when.strftime('%d %H:%M')}Z / {local.strftime('%d %H:%M')}"
    return f"{lane.name:<17}{state:<10}{progress:<24}{shown:<15}{eta}"


def render(board: Board, now: datetime, polled: dict, history) -> str:
    local = now.astimezone(timezone(timedelta(hours=board.report_utc_offset_hours)))
    out = [f"{now.strftime('%d %b %H:%M')}Z / {local.strftime('%H:%M')} local", ""]
    out += describe_local(board)
    out += ["", f"{'LANE':<17}{'STATE':<10}{'PROGRESS':<24}{'RATE':<15}ETA (UTC / local)"]
    out += [describe_lane(board, lane, polled, history, now) for lane in board.lanes]
    out.append("")
    for host in board.hosts:
        _rows, gpu, error = polled[host]
        out.append(f"{host + ' GPU':<17}{gpu or error}")
    return "\n".join(out)


def main() -> None:
    settings = Settings()
    if settings.board is None:
        raise SystemExit("config.toml has no [board]: the lane table lives there, not in this module")
    board = settings.board
    now = datetime.now(timezone.utc).replace(microsecond=0)
    polled = {host: poll_host(board, host) for host in board.hosts}
    print(render(board, now, polled, read_history(board.history)))
    append_history(board.history, now, {lane.name: polled[lane.host][0].get(lane.name, (-1, ""))[0]
                                        for lane in board.lanes if not polled[lane.host][2]})


if __name__ == "__main__":
    main()
