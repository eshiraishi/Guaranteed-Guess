import json, pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("ahmedheakl/gg-armv8-O2", revision="0444a378f384300187c7f43914fc986a81ff051a")
t = pq.read_table("gg-bench-bringup-O2.parquet").to_pydict()
plen = {f: len(tok(x).input_ids) for f, x in zip(t["file"], t["x86"])}
for m in ("released", "beam8"):
    rows = [json.loads(l) for l in open(f"bringup_O2_{m}.jsonl")]
    done = {r["file"] for r in rows}
    left = sorted(plen[f] for f in plen if f not in done)
    el = sum(r["seconds"] for r in rows)
    slow = max(rows, key=lambda r: r["seconds"])
    print(m, f"done {len(rows)}/66 in {el / 3600:.1f} h; slowest so far {slow['seconds'] / 60:.0f} min (prompt {slow['prompt_tokens']});",
          f"remaining prompt tokens: {left}")
