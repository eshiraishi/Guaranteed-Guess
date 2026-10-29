"""Exact (normalized) x86 overlap inside valid MBPP-C and between MBPP-C and HumanEval-C."""
import sys
from collections import defaultdict

import polars as pl

sys.path.insert(0, sys.argv[3])
from contam_probe_norm import normalize  # noqa: E402

mbpp = pl.read_parquet(sys.argv[1]).filter(ref_verdict="pass", neg_verdict="link_err")
he = pl.read_parquet(sys.argv[2])
groups = defaultdict(list)
for tid, x in mbpp.select("task_id", "x86").iter_rows():
    groups["\n".join(normalize(x))].append(tid)
dups = [v for v in groups.values() if len(v) > 1]
print("valid MBPP-C:", mbpp.height, "| distinct normalized x86:", len(groups), "| dup groups:", len(dups), dups[:15])
cross = [(f, groups[k]) for f, x in he.select("file", "x86").iter_rows() if (k := "\n".join(normalize(x))) in groups]
print("HumanEval-C functions identical to an MBPP-C training function:", len(cross), cross)
