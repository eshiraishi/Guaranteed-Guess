"""Show the region of a Stack file that shares canonical 13-grams with a probe. usage: show.py SCRATCH GGRPO probe file [probe file ...]"""
import re, sys, glob, polars as pl
S, G, args = sys.argv[1], sys.argv[2], sys.argv[3:]
TOK = r"[A-Za-z_]\w*|\d[\w.]*|->|\+\+|--|<<=?|>>=?|[<>=!]=|&&|\|\||[-+*/%&|^]=|[^\s\w]"
lib = open(f"{S}/scripts/neardup_lib.py").read(); KW = set(re.search(r'KW = """(.*?)"""', lib, re.S).group(1).split())
def clean(c):
    for p, r in [(r'"(?:\\.|[^"\\\n])*"', " S "), (r"'(?:\\.|[^'\\\n])*'", " C "), (r"(?s)/\*.*?\*/", " "), (r"//[^\n]*", " "), (r"(?m)^[ \t]*#[^\n]*", " ")]:
        c = re.sub(p, r, c)
    return c
def toks(c):
    c = clean(c); m = list(re.finditer(TOK, c))
    return c, [(x.group() if not re.match(r"[A-Za-z_]", x.group()) or x.group() in KW else "I") for x in m], m
he = pl.read_parquet(f"{G}/data/raw/asmbench_source.parquet").select(probe=pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1), code="code")
for probe, f in zip(args[::2], args[1::2]):
    pc = he.filter(pl.col("probe") == probe)["code"][0] if probe.startswith("humaneval") else "\n".join(open(x).read() for x in sorted(glob.glob(f"{S}/bringup_src/repo/{probe[8:]}/*.c")))
    sc = pl.scan_parquet(f"{S}/stackc/*.parquet").filter(pl.col("file") == f).collect()["data"][0]
    _, pt, _ = toks(pc); c, st, sm = toks(sc)
    P = {tuple(pt[i:i + 13]) for i in range(len(pt) - 12)}
    hit = [i for i in range(len(st) - 12) if tuple(st[i:i + 13]) in P]
    a, b = sm[hit[0]].start(), sm[min(hit[-1] + 12, len(sm) - 1)].end()
    body = re.sub(r"\n\s*\n+", "\n", re.sub(r"(?m)^[ \t]*#[^\n]*", "", pc)).strip()
    print(f"######## {probe}\n{body[:900]}\n-------- {f} [{len(hit)} shared windows]\n{re.sub(r'\n\s*\n+', chr(10), c[a:b])[:900]}\n")
