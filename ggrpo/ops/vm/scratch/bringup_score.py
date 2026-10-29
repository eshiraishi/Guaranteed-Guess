"""Upstream BringUp eval (eval/bringup-bench Makefile, `make run-tests TARGET=host`) on a predictions file.
Fresh copy of the bench per call; only benchmarks with a prediction are run (TARGET_BMARKS), because a missing .s
makes the Makefile fall back to compiling the original C (a silent pass). Missing ones count as failures.
One `make` per benchmark: the Makefile's `timeout 30s` marks a runaway program FAILURE but can leave it running
(seen: gcd-list, 1h+ at 100% CPU), which would slow later benchmarks and keep the output pipe open, so leftovers are
killed after each benchmark (output goes to a file so make's exit is not blocked). Program output can be
non-UTF-8, hence errors="replace".
Usage: bringup_score.py preds.json out.jsonl   (preds: save_preds.py format {"files": [...], "pred": [...]})"""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
src = Path.home() / "bringup-bench-clean"
d = json.load(open(sys.argv[1]))
preds = {f.removesuffix(".s"): p for f, p in zip(d["files"], d["pred"])}
bmarks = re.search(r"^BMARKS = (.*)$", (src / "Makefile").read_text(), re.M).group(1).split()
ok, bad, leftovers = set(), set(), []
with tempfile.TemporaryDirectory() as tmp:
    bb = Path(tmp, "bb"); shutil.copytree(src, bb)
    for b, p in preds.items():
        (bb / b / f"{b}.s").write_text(p)
    run = [b for b in bmarks if b in preds]
    for b in run:
        log = Path(tmp, f"{b}.log")  # a file, not a pipe: a leftover program would hold a pipe open
        with log.open("w") as f:
            subprocess.run(["make", "run-tests", "TARGET=host", f"TARGET_BMARKS={b}"], cwd=bb, stdout=f, stderr=subprocess.STDOUT)
        out = log.read_text(errors="replace")
        ok |= set(re.findall(r"SUCCESS: (\S+)", out)); bad |= set(re.findall(r"FAILURE: (\S+)", out))
        if subprocess.run(["pkill", "-9", "-f", f"^\\./{re.escape(b)}\\.host"]).returncode == 0:
            leftovers.append(b)
assert ok | bad == set(run) and not ok & bad, (len(ok | bad), len(run))
rows = [{"bench": b, "verdict": "pass" if b in ok else "fail" if b in bad else "missing", "left_running": b in leftovers}
        for b in bmarks]
Path(sys.argv[2]).write_text("".join(json.dumps(r) + "\n" for r in rows))
n65 = [r for r in rows if r["bench"] != "satomi"]
print(json.dumps({"preds": len(preds), "pass": len(ok), "of_66": len(bmarks), "pass_of_65_excl_satomi": sum(r["verdict"] == "pass" for r in n65),
                  "missing": [r["bench"] for r in rows if r["verdict"] == "missing"], "left_running": leftovers}))
