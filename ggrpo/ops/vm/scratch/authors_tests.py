"""Authors' asmbench_source (code+test per eval/problem{n}/code.c) vs the official gg-bench rows, in the VM toolchain.
1) recompile their code (O0/O2, x86/arm) and compare byte-for-byte with the official asm;
2) official reference arm and an empty prediction against THEIR tests;
3) step-0 predictions against THEIR tests."""
import json, subprocess, tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import polars as pl
from ggrpo.harness import check
from ggrpo.settings import Settings

D = Path.home() / "scratch/data"

def compile_asm(src, opt, target):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "code.c").write_text(src)
        arch = ["-target", "x86_64-apple-darwin"] if target == "x86" else ["-arch", "arm64"]
        r = subprocess.run(["clang", "-S", *arch, "code.c", "-o", "code.s", f"-{opt}"], cwd=d, capture_output=True)
        return Path(d, "code.s").read_text() if r.returncode == 0 else None

def job(r):
    out = {"file": r["file"], "opt": r["opt"],
           "x86_same": compile_asm(r["code"], r["opt"], "x86") == r["x86"],
           "arm_same": compile_asm(r["code"], r["opt"], "arm") == r["arm"],
           "ref": check(r["arm"], r["test"]).verdict, "empty": check("", r["test"]).verdict}
    out["pred"] = check(r["pred"], r["test"]).verdict if r.get("pred") is not None else None
    return out

if __name__ == "__main__":
    s = Settings()
    src = pl.read_parquet(D / "asmbench_source.parquet")
    preds = (pl.read_ndjson(s.data.step0_predictions)
             .select(file=pl.format("eval/problem{}/code.c", pl.col("task_id").str.extract(r"(\d+)").cast(pl.Int64) + 1),
                     pred=pl.col("raw").str.extract(s.prompt.extract, 1)))
    rows = pl.concat([pl.read_parquet(D / f"official_{o}.parquet").select("file", "x86", "arm").with_columns(opt=pl.lit(o))
                      for o in ("O0", "O2")])
    rows = rows.join(src, on="file", how="left", validate="m:1").join(
        preds.with_columns(opt=pl.lit("O2")), on=["file", "opt"], how="left")
    assert rows["code"].null_count() == 0 and rows.height == 328, rows.height
    with Pool(4) as p:
        res = pl.DataFrame(p.map(job, rows.to_dicts()), infer_schema_length=None)
    res.write_ndjson(Path.home() / "scratch/authors_tests.jsonl")
    for opt in ("O0", "O2"):
        r = res.filter(opt=opt)
        print(opt, json.dumps({"x86_same": r["x86_same"].sum(), "arm_same": r["arm_same"].sum(), "n": r.height,
                               "ref": Counter(r["ref"]), "empty": Counter(r["empty"])}))
        print("  not identical:", r.filter(~(pl.col("x86_same") & pl.col("arm_same")))["file"].to_list())
        print("  ref not pass:", r.filter(pl.col("ref") != "pass")["file"].to_list())
    p = res.filter(opt="O2")
    print("step0 preds vs authors' tests:", json.dumps(Counter(p["pred"])), "pass", (p["pred"] == "pass").sum(), "/", p.height)
