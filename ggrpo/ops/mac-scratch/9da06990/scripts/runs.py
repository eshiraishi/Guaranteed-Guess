"""Longest contiguous run of shared canonical 13-grams between each BringUp/HumanEval probe and its top-3 Stack files."""
import re, sys, glob, polars as pl
S, G = sys.argv[1], sys.argv[2]
exec(open(f"{S}/scripts/show.py").read().split("he = pl.read_parquet")[0].split("S, G, args")[0])
TOK = r"[A-Za-z_]\w*|\d[\w.]*|->|\+\+|--|<<=?|>>=?|[<>=!]=|&&|\|\||[-+*/%&|^]=|[^\s\w]"
KW = set(re.search(r'KW = """(.*?)"""', open(f"{S}/scripts/neardup_lib.py").read(), re.S).group(1).split())
def clean(c):
    for p, r in [(r'"(?:\\.|[^"\\\n])*"', " S "), (r"'(?:\\.|[^'\\\n])*'", " C "), (r"(?s)/\*.*?\*/", " "), (r"//[^\n]*", " "), (r"(?m)^[ \t]*#[^\n]*", " ")]:
        c = re.sub(p, r, c)
    return c
toks = lambda c: [(t if not re.match(r"[A-Za-z_]", t) or t in KW else "I") for t in re.findall(TOK, clean(c))]
res = pl.read_parquet(f"{S}/neardup_c.parquet").filter(pl.col("probe").str.contains("bringup|humaneval"))
stack = pl.scan_parquet(f"{S}/stackc/*.parquet").filter(pl.col("file").is_in(res["file"].unique().implode())).collect()
src = dict(stack.iter_rows())
he = dict(pl.read_parquet(f"{G}/data/raw/asmbench_source.parquet").select(pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1), "code").iter_rows())
out = []
for probe, f in res.select("probe", "file").iter_rows():
    pc = he[probe] if probe.startswith("humaneval") else "\n".join(open(x).read() for x in sorted(glob.glob(f"{S}/bringup_src/repo/{probe[8:]}/*.c")))
    pt, st = toks(pc), toks(src[f])
    P = {tuple(pt[i:i + 13]) for i in range(len(pt) - 12)}
    best = cur = 0
    for i in range(len(st) - 12):
        cur = cur + 1 if tuple(st[i:i + 13]) in P else 0
        best = max(best, cur)
    out.append((probe, f, best + 12 if best else 0, len(pt)))
df = pl.DataFrame(out, schema=["probe", "file", "run_tokens", "probe_tokens"], orient="row").sort("run_tokens", descending=True).group_by("probe", maintain_order=True).first()
df.write_parquet(f"{S}/neardup_runs.parquet")
for k in ("bringup", "humaneval"):
    print(k, " ".join(f"{r[0].split('/')[1]}={r[2]}/{r[3]}" for r in df.filter(pl.col("probe").str.starts_with(k)).iter_rows() if r[2] >= 40))
