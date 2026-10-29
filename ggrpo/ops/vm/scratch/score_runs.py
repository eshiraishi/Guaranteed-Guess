"""Score generation runs ({file, raw, prompt_tokens, new_tokens, error} per line) in the VM, two ways:
ours = LLM4Decompile driver (data/eval.jsonl; humaneval/k <-> eval/problem{k+1}), authors = asmbench_source test.
Usage (from ~/ggrpo): score_runs.py run1.jsonl [run2.jsonl ...]"""
import json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import polars as pl
from ggrpo.harness import check
from ggrpo.settings import Settings

def job(r):
    return {"file": r["file"], "ours": check(r["asm"], r["driver"]).verdict, "authors": check(r["asm"], r["test"]).verdict}

if __name__ == "__main__":
    s = Settings()
    # LLM4Decompile drivers (data/eval.jsonl switched to the authors' tests at 11:24); 158-valid set from the step-0 VM check
    ours = pl.read_ndjson("data/eval_llm4decompile.jsonl").select(
        "driver", file=pl.format("eval/problem{}/code.c", pl.col("task_id").str.extract(r"(\d+)").cast(pl.Int64) + 1))
    valid = set(pl.read_ndjson("data/raw/step0_ref_empty_O2.jsonl").filter(kind="ref", passed=True).select(
        pl.format("eval/problem{}/code.c", pl.col("task_id").str.extract(r"(\d+)").cast(pl.Int64) + 1)).to_series())
    test = pl.read_parquet(Path.home() / "scratch/data/asmbench_source.parquet").select("file", "test")
    for run in sys.argv[1:]:
        r = pl.read_ndjson(run, infer_schema_length=None)
        rows = (r.join(ours, on="file", how="left", validate="1:1").join(test, on="file", how="left", validate="1:1")
                .with_columns(asm=pl.col("raw").str.extract(s.prompt.extract, 1)))
        assert rows.height == 164 and rows["driver"].null_count() == 0 and rows["test"].null_count() == 0, (run, rows.height)
        with Pool(4) as p:
            v = pl.DataFrame(p.map(job, rows.select("file", "asm", "driver", "test").to_dicts()))
        v.join(r.select("file", "new_tokens", "prompt_tokens"), on="file").write_ndjson(Path(run).with_suffix(".scored.jsonl"))
        print(json.dumps({"run": Path(run).name,
                          "authors_164": int((v["authors"] == "pass").sum()),
                          "ours_164": int((v["ours"] == "pass").sum()),
                          "ours_158": int(v.filter(pl.col("file").is_in(valid))["ours"].eq("pass").sum()),
                          "hit_cap": int((r["prompt_tokens"] + r["new_tokens"] >= 32000).sum()),
                          "errors": Counter(r["error"].drop_nulls().to_list()) if "error" in r.columns else {},
                          "authors_verdicts": Counter(v["authors"].to_list())}))
