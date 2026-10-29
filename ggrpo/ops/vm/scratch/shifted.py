"""Score the 5 shifted HumanEval slots (k holds problem k+1's function) against problem k+1's tests."""
import json, polars as pl
from ggrpo.harness import check
from ggrpo.settings import Settings
s = Settings()
ev = {r["task_id"]: r for r in pl.read_ndjson(s.data.eval).to_dicts()}
pred = {r["task_id"]: r["raw"] for r in pl.read_ndjson(s.data.step0_predictions).to_dicts()}
extract = lambda raw: pl.Series([raw]).str.extract(s.prompt.extract, 1)[0]
for k in [1, 101, 125, 137, 162]:
    a, b = f"humaneval/{k}", f"humaneval/{k+1}"
    print(json.dumps({"slot": a, "ref_vs_own": check(ev[a]["arm_ref"], ev[a]["driver"]).verdict,
                      "ref_vs_next": check(ev[a]["arm_ref"], ev[b]["driver"]).verdict,
                      "pred_vs_own": check(extract(pred[a]), ev[a]["driver"]).verdict,
                      "pred_vs_next": check(extract(pred[a]), ev[b]["driver"]).verdict,
                      "next_pred_vs_next": check(extract(pred[b]), ev[b]["driver"]).verdict}))
