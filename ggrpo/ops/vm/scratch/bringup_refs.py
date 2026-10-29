"""Upstream BringUp eval (eval/bringup-bench Makefile, `make run-tests TARGET=host`) on the official reference arm.
Usage: bringup_refs.py O0|O2  (writes <b>/<b>.s for every dataset row, like save_preds.py)."""
import subprocess, sys, re
from pathlib import Path
import polars as pl
opt = sys.argv[1]
bb = Path.home() / "bringup-bench"
df = pl.read_parquet(Path.home() / f"scratch/data/bringup_{opt}.parquet").with_columns(pl.col("file").str.replace(r"\.s$", ""))  # save_preds.py: file.replace(".s", "")
bmarks = re.search(r"^BMARKS = (.*)$", (bb / "Makefile").read_text(), re.M).group(1).split()
print("dataset rows", df.height, "| BMARKS", len(bmarks), "| in dataset not in BMARKS:", sorted(set(df["file"]) - set(bmarks)),
      "| in BMARKS not in dataset:", sorted(set(bmarks) - set(df["file"])))
for r in df.iter_rows(named=True):
    (bb / r["file"] / f"{r['file']}.s").write_text(r["arm"])
out = subprocess.run(["make", "run-tests", "TARGET=host"], cwd=bb, capture_output=True, text=True).stdout
print(opt, "failures:", re.findall(r"FAILURE: (\S+)", out)); print("\n".join(out.splitlines()[-5:]))
