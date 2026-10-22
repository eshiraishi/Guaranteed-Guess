"""Export the official HumanEval-C rows (gg-official, one opt level) from the existing parquet to JSONL
for the VM. Read-only on rl/artifacts/data."""
import json, sys
import pandas as pd
opt = sys.argv[1]; out = sys.argv[2]
df = pd.read_parquet("rl/artifacts/data/humaneval_armv8.parquet")
df = df[(df.x86_origin == "gg-official") & (df.opt == opt)]
with open(out, "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({"task_id": r.task_id, "opt": r.opt, "x86": r.x86, "arm_ref": r.arm_ref,
                            "support_src": json.loads(r.support_src), "cflags": json.loads(r.cflags)}) + "\n")
print(len(df), "rows ->", out)
