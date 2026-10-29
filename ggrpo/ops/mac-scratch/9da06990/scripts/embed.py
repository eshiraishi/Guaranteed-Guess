"""Embedding similarity of MBPP (train/dev) vs HumanEval problem statements; top-3 MBPP per HumanEval."""
import json, sys, numpy as np, polars as pl
from fastembed import TextEmbedding
S, G = sys.argv[1], sys.argv[2]
used = {json.loads(l)["task_id"] for f in ("train", "dev") for l in open(f"{G}/data/{f}.jsonl")}
m = (pl.concat([pl.read_parquet(f"{S}/probs/mbpp_{s}.parquet") for s in ("prompt", "test", "train", "validation")])
     .select(probe=pl.format("mbpp/{}", pl.col("task_id") - 1), t=pl.col("text").str.replace(r"^Write (a |an )?(python )?function to ", ""))
     .filter(pl.col("probe").is_in(used)).sort("probe"))
h = pl.read_parquet(f"{S}/probs/humaneval.parquet").select(
    probe=pl.format("humaneval/{}", pl.col("task_id").str.extract(r"(\d+)$")), ep="entry_point",
    t=pl.col("prompt").str.extract(r'(?s)"""(.*?)(?:>>>|For example|Example|""")', 1)
      .fill_null(pl.col("prompt").str.extract(r"(?s)'''(.*?)(?:>>>|For example|Example|''')", 1)).str.replace_all(r"\s+", " ").str.strip_chars())
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=f"{S}/emb")
E = lambda xs: (lambda a: a / np.linalg.norm(a, axis=1, keepdims=True))(np.array(list(model.embed(xs))))
sim = E(h["t"].to_list()) @ E(m["t"].to_list()).T
rows = [(h["probe"][i], h["ep"][i], m["probe"][j], float(sim[i, j]), m["t"][j]) for i in range(len(h)) for j in map(int, np.argsort(-sim[i])[:3])]
pl.DataFrame(rows, schema=["he", "ep", "mbpp", "cos", "mtext"], orient="row").write_parquet(f"{S}/sameprob_embed.parquet")
for he, ep, mb, c, t in rows:
    if c >= 0.80: print(f"{he[10:]:>3} {ep[:22]:<22} {c:.2f} {mb:<9} {t[:90]}")
