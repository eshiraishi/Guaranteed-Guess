"""Per row: RTX 5000 Ada vs Blackwell beam8 (seconds, ratio, peak memory, identical output?).
Usage: compare_gpu.py cluster.jsonl blackwell.jsonl"""
import json, sys
new = [json.loads(l) for l in open(sys.argv[1])]
old = {r["file"]: r for r in map(json.loads, open(sys.argv[2]))}
for r in new:
    o = old.get(r["file"])
    ratio = round(r["seconds"] / o["seconds"], 2) if o and o["seconds"] else None
    print(json.dumps({"file": r["file"], "prompt": r["prompt_tokens"], "new": r["new_tokens"], "s": r["seconds"],
                      "blackwell_s": o and o["seconds"], "ratio": ratio, "peak_gib": r.get("peak_gib"), "error": r["error"],
                      "same_as_blackwell": o["raw"] == r["raw"] if o else None}))
both = [r for r in new if r["file"] in old]
if both:
    print("total ratio", round(sum(r["seconds"] for r in both) / sum(old[r["file"]]["seconds"] for r in both), 2),
          "identical", sum(old[r["file"]]["raw"] == r["raw"] for r in both), "of", len(both))
