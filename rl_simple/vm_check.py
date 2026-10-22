"""Runs INSIDE the macOS 13 VM (stdlib only). Usage: vm_check.py tasks.jsonl [preds.jsonl] out.jsonl
Without preds: checks the official reference ARM and an empty prediction.
With preds ({"task_id","pred"} per line): scores the predictions."""
import json, sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_copy"))
from harness import Harness
h = Harness(run_timeout=10)
tasks = [json.loads(l) for l in open(sys.argv[1])]
if len(sys.argv) == 4:
    preds = {}
    for l in open(sys.argv[2]):
        d = json.loads(l); preds[d["task_id"]] = d["pred"]
    jobs = [(t, "pred", preds.get(t["task_id"], "")) for t in tasks]
    out = sys.argv[3]
else:
    jobs = [(t, "ref", t["arm_ref"]) for t in tasks] + [(t, "empty", "") for t in tasks]
    out = sys.argv[2]
cnt = collections.Counter()
with open(out, "w") as f:
    for t, kind, asm in jobs:
        task = {"support_src": t["support_src"], "cflags": t["cflags"], "ldflags": []}
        r = h.check(asm, task).to_dict()
        r.update(task_id=t["task_id"], kind=kind)
        f.write(json.dumps(r) + "\n"); cnt[(kind, r["verdict"])] += 1
for k, v in sorted(cnt.items()): print(k, v)
