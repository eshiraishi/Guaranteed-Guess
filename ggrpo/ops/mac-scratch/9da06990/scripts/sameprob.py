"""Candidate same-problem pairs between MBPP-C (train/dev) and HumanEval-C (eval), for manual review.

Ranks pairs by TF-IDF cosine of the original problem statements and by canonical C-token 5-gram Jaccard.
usage: sameprob.py SCRATCH GGRPO
"""
import importlib.util, json, sys
import polars as pl

S, G = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("nd", f"{S}/scripts/neardup_lib.py")
nd = importlib.util.module_from_spec(spec); spec.loader.exec_module(nd)

used = {json.loads(l)["task_id"] for f in ("train", "dev") for l in open(f"{G}/data/{f}.jsonl")}
mbpp_text = pl.concat([pl.read_parquet(f"{S}/probs/mbpp_{s}.parquet") for s in ("prompt", "test", "train", "validation")])
mbpp = (pl.read_parquet(f"{G}/data/raw/decompile_eval_mbpp.parquet").sort("index").with_row_index("rank")
        .join(mbpp_text.select("task_id", "text", py="code"), left_on=pl.col("rank").cast(pl.Int64) + 1, right_on="task_id")
        .select(probe=pl.format("mbpp/{}", "rank"), text="text", code="func", py="py").filter(pl.col("probe").is_in(used)))
he = (pl.read_parquet(f"{S}/probs/humaneval.parquet")
      .select(probe=pl.format("humaneval/{}", pl.col("task_id").str.extract(r"(\d+)$")),
              text=pl.col("prompt").str.extract(r'(?s)"""(.*?)(?:>>>|For example|Example|""")', 1)
                   .fill_null(pl.col("prompt").str.extract(r"(?s)'''(.*?)(?:>>>|For example|Example|''')", 1)))
      .join(pl.read_parquet(f"{G}/data/raw/asmbench_source.parquet").select(
          probe=pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1), code="code"),
          on="probe"))
docs = pl.concat([mbpp.select("probe", "text", "code"), he.select("probe", "text", "code")])

# TF-IDF over problem statements
STOP = set("the a an of to and in is are be for that with given from by or as it its this on at return returns write function "
           "list number numbers string strings which if each all".split())
words = (docs.select("probe", w=pl.col("text").str.to_lowercase().str.extract_all(r"[a-z]{3,}")).explode("w")
         .filter(~pl.col("w").is_in(list(STOP))).group_by("probe", "w").len("tf"))
idf = words.group_by("w").len("df").with_columns(idf=(docs.height / pl.col("df")).log())
vec = (words.join(idf, on="w").with_columns(v=pl.col("tf") * pl.col("idf"))
       .with_columns(v=pl.col("v") / (pl.col("v") ** 2).sum().sqrt().over("probe")))
m, h = vec.filter(pl.col("probe").str.starts_with("mbpp")), vec.filter(pl.col("probe").str.starts_with("humaneval"))
text_sim = m.join(h, on="w", suffix="_he").group_by("probe", "probe_he").agg(text=(pl.col("v") * pl.col("v_he")).sum())

# canonical C 5-gram Jaccard
sh = nd.shingles(docs.with_row_index("id").select("id", tok=nd.c_tokens(pl.col("code"))), 5, nd.canon).join(
    docs.with_row_index("id").select("id", "probe"), on="id").drop("id")
n = sh.group_by("probe").len("n")
ms, hs = sh.filter(pl.col("probe").str.starts_with("mbpp")), sh.filter(pl.col("probe").str.starts_with("humaneval"))
code_sim = (ms.join(hs, on="h", suffix="_he").group_by("probe", "probe_he").len("inter")
            .join(n, on="probe").join(n.rename({"probe": "probe_he", "n": "n_he"}), on="probe_he")
            .select("probe", "probe_he", code=pl.col("inter") / (pl.col("n") + pl.col("n_he") - pl.col("inter"))))

pairs = text_sim.join(code_sim, on=["probe", "probe_he"], how="full", coalesce=True).fill_null(0.0)
cand = pl.concat([pairs.sort("text", descending=True).head(60), pairs.sort("code", descending=True).head(30)]).unique(
    ["probe", "probe_he"]).sort("text", descending=True)
cand = cand.join(mbpp.select("probe", mbpp_text="text"), on="probe").join(
    he.select(probe_he="probe", he_text=pl.col("text").str.replace_all(r"\s+", " ").str.strip_chars()), on="probe_he")
cand.write_parquet(f"{S}/sameprob_candidates.parquet")
pairs.write_parquet(f"{S}/sameprob_pairs.parquet")
for r in cand.iter_rows(named=True):
    print(f"{r['probe']:>9} ~ {r['probe_he']:<13} text={r['text']:.2f} code={r['code']:.2f}\n"
          f"   M: {r['mbpp_text'][:150]}\n   H: {r['he_text'][:150]}")
