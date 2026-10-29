"""gen_bringup.py JSONL -> save_preds.py-style JSON {"files", "pred"}; extraction as eval/eval_bringupbench.py
(decoded.split("```armasm\n")[-1].split("```")[0]); rows with an error (context_overflow/oom) get pred "" (fails to link)."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
assert len({r["file"] for r in rows}) == len(rows) == 66, len(rows)
json.dump({"files": [r["file"] for r in rows],
           "pred": ["" if r["error"] else r["raw"].split("```armasm\n")[-1].split("```")[0] for r in rows]}, open(sys.argv[2], "w"))
print(sys.argv[1], "errors:", {e: sum(r["error"] == e for r in rows) for e in {r["error"] for r in rows} if e},
      "hit 32k total:", sum(r["prompt_tokens"] + r["new_tokens"] >= 32000 for r in rows))
