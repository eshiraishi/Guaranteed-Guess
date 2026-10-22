"""Export MBPP-C jobs (same selection and driver as rl/data/build_humaneval.py) for the VM. Read-only on the repo."""
import json, os, sys
sys.path.insert(0, "rl/data")
from datasets import load_from_disk
from build_humaneval import driver_source
opt = sys.argv[1]; out = sys.argv[2]
ds = load_from_disk("data_cache/decompile-eval/mbpp")
rows = [dict(r) for r in ds if r["language"] == "c" and r["opt"] == "O0"]
rows.sort(key=lambda r: int(r["index"]))
n = 0
with open(out, "w") as f:
    for k, r in enumerate(rows):
        drv = driver_source(r)
        f.write(json.dumps({"task_id": f"mbpp/{k}", "source": "mbpp", "opt": opt,
                            "code": r["func_dep"] + "\n" + r["func"],
                            "support_src": {"test.c": drv} if drv else None, "cflags": []}) + "\n"); n += 1
print(n, "jobs ->", out)
