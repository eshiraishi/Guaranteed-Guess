import json, collections
import pandas as pd
TOOLCHAIN = open("rl_simple/step0_toolchain.txt").read().strip()
rows = [json.loads(l) for l in open("rl_simple/mbpp_O2_vm.jsonl")]
rows.sort(key=lambda r: int(r["task_id"].split("/")[1]))
print("total", len(rows)); print("drop reasons", collections.Counter(r["drop"] for r in rows))
print("ref verdicts", collections.Counter(r["ref_verdict"] for r in rows)); print("neg", collections.Counter(r["neg_verdict"] for r in rows))
keep = [r for r in rows if r["drop"] is None]
df = pd.DataFrame([{**{k: r[k] for k in ["task_id","source","opt","x86","arm_ref","ref_verdict","neg_verdict","ref_stderr"]},
                    "x86_origin": "regenerated", "support_src": json.dumps(r["support_src"]), "cflags": json.dumps(r["cflags"]),
                    "toolchain": TOOLCHAIN} for r in keep])
df.to_parquet("rl_simple/mbpp_armv8_O2_ventura.parquet")
ok = df[df.ref_verdict == "pass"]
print("rows", len(df), "usable (ref pass, neg not pass):", len(ok[ok.neg_verdict != "pass"]))
print("failed:", df[df.ref_verdict != "pass"][["task_id","ref_verdict"]].values.tolist())
old = pd.read_parquet("rl/artifacts/data/mbpp_armv8.parquet"); old = old[old.opt == "O2"]
op = set(old[old.ref_verdict == "pass"].task_id); np_ = set(ok.task_id)
print("old pass", len(op), "new pass", len(np_), "old-only", sorted(op - np_), "new-only", sorted(np_ - op))
