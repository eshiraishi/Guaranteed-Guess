"""Feasibility probe: exact and 10-instruction-window overlap between benchmark x86 and one GG SFT shard."""
import re
import sys
import time

import polars as pl

K = 10
SYM = re.compile(r"\b_[A-Za-z_][A-Za-z0-9_.$]*")


def normalize(asm: str) -> list[str]:
    out = []
    for line in asm.splitlines():
        line = line.split("##")[0].rstrip()
        s = line.strip()
        if not s or (s.startswith(".") and not s.startswith(".L")):
            continue
        out.append(SYM.sub("_S", s))
    return out


def windows(lines: list[str]) -> set[int]:
    return {hash("\n".join(lines[i : i + K])) for i in range(max(1, len(lines) - K + 1))}


shard, mbpp, humaneval = sys.argv[1:4]
bench = pl.concat([
    pl.read_parquet(mbpp).select(pl.lit("mbpp").alias("set"), pl.col("task_id").cast(pl.String).alias("id"), "x86"),
    pl.read_parquet(humaneval).select(pl.lit("humaneval").alias("set"), pl.col("file").alias("id"), "x86"),
])
t = time.time()
bench_norm = {(r["set"], r["id"]): normalize(r["x86"]) for r in bench.iter_rows(named=True)}
exact = {"\n".join(v): k for k, v in bench_norm.items()}
bwin = {k: windows(v) for k, v in bench_norm.items()}
allwin = set().union(*bwin.values())

sft = pl.read_parquet(shard, columns=["x86", "file"])
hits_exact, seen = [], set()
for x86, f in sft.iter_rows():
    n = normalize(x86)
    j = "\n".join(n)
    if j in exact:
        hits_exact.append((exact[j], f))
    seen |= windows(n) & allwin
print(f"shard rows {sft.height}, bench {len(bench_norm)}, {time.time() - t:.0f}s")
print("exact matches:", len(hits_exact))
for h in hits_exact[:20]:
    print("  ", h)
for s in ("mbpp", "humaneval"):
    fr = [len(w & seen) / len(w) for (bs, _), w in bwin.items() if bs == s]
    print(s, "funcs with >=50% of 10-instr windows seen:", sum(x >= 0.5 for x in fr), "/", len(fr),
          "| 100%:", sum(x == 1 for x in fr))

ln = {k: len(v) for k, v in bench_norm.items()}
print("bench instr count quantiles:", sorted(ln.values())[len(ln) // 10], sorted(ln.values())[len(ln) // 2])
hit_keys = {k for k, _ in hits_exact}
print("exact-matched bench funcs and their instr counts:", sorted((ln[k], k[1]) for k in hit_keys))
win_full = [(ln[k], k[1]) for k, w in bwin.items() if w <= seen]
print("100%-window funcs:", sorted(win_full))
