"""Print the longest shared canonical-13-gram run between probe and file. usage: runshow.py SCRATCH GGRPO probe [probe...]"""
import re, sys, glob, polars as pl
S, G, probes = sys.argv[1], sys.argv[2], sys.argv[3:]
TOK = r"[A-Za-z_]\w*|\d[\w.]*|->|\+\+|--|<<=?|>>=?|[<>=!]=|&&|\|\||[-+*/%&|^]=|[^\s\w]"
KW = set(re.search(r'KW = """(.*?)"""', open(f"{S}/scripts/neardup_lib.py").read(), re.S).group(1).split())
def clean(c):
    for p, r in [(r'"(?:\\.|[^"\\\n])*"', " S "), (r"'(?:\\.|[^'\\\n])*'", " C "), (r"(?s)/\*.*?\*/", " "), (r"//[^\n]*", " "), (r"(?m)^[ \t]*#[^\n]*", " ")]:
        c = re.sub(p, r, c)
    return c
runs = pl.read_parquet(f"{S}/neardup_runs.parquet")
he = dict(pl.read_parquet(f"{G}/data/raw/asmbench_source.parquet").select(pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1), "code").iter_rows())
for probe in probes:
    f = runs.filter(pl.col("probe") == probe)["file"][0]
    pc = he[probe] if probe.startswith("humaneval") else "\n".join(open(x).read() for x in sorted(glob.glob(f"{S}/bringup_src/repo/{probe[8:]}/*.c")))
    sc = clean(pl.scan_parquet(f"{S}/stackc/*.parquet").filter(pl.col("file") == f).collect()["data"][0])
    m = list(re.finditer(TOK, sc)); st = [(x.group() if not re.match(r"[A-Za-z_]", x.group()) or x.group() in KW else "I") for x in m]
    pt = [(t if not re.match(r"[A-Za-z_]", t) or t in KW else "I") for t in re.findall(TOK, clean(pc))]
    P = {tuple(pt[i:i + 13]) for i in range(len(pt) - 12)}
    best, cur, end = 0, 0, 0
    for i in range(len(st) - 12):
        cur = cur + 1 if tuple(st[i:i + 13]) in P else 0
        if cur > best: best, end = cur, i
    a, b = m[end - best + 1].start(), m[end + 12].end()
    print(f"######## {probe} vs {f}: run {best + 12} tokens\n{re.sub(chr(10)+r'\s*'+chr(10)+'+', chr(10), sc[a:b])[:700]}\n")
