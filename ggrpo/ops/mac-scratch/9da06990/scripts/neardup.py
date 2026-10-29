"""Near-duplicate probe of HumanEval-C, BringUp and MBPP-C against GG's SFT data.

C level: canonical-token 13-gram containment against the C source of every SFT Stack file.
x86 level: normalized-instruction 10-gram containment of each probe function against every SFT row (both parts).
Shingles shared by many probes or many corpus docs are dropped as boilerplate.

usage: neardup.py SCRATCH GGRPO [LIMIT]   (LIMIT: corpus files per level, for a smoke test)
"""
import functools, glob, operator, os, sys, time
import polars as pl

S, G = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
K_C, K_ASM, MAX_PROBES, MAX_DF, BATCH = 13, 10, 20, 1000, 20_000
MULT = [pl.lit((0x9E3779B97F4A7C15 * (2 * i + 1)) % 2**64 | 1, dtype=pl.UInt64) for i in range(16)]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- shingling
def shingles(df, k, transform=None, keep=None):
    """df(id, tok: list[str]) -> distinct (id, h) of k-token windows; `keep` restricts to known hashes."""
    x = (df.with_columns(pos=pl.int_ranges(0, pl.col("tok").list.len()), n=pl.col("tok").list.len())
         .filter(pl.col("n") >= k).explode(["tok", "pos"]))
    x = x.with_columns(t=(transform(pl.col("tok")) if transform else pl.col("tok")).hash(7))
    h = functools.reduce(operator.add, [pl.col("t").shift(-i) * MULT[i] for i in range(k)])
    x = x.with_columns(h=h).filter(pl.col("pos") <= pl.col("n") - k)
    if keep is not None:
        x = x.filter(pl.col("h").is_in(keep))
    return x.select("id", "h").unique()

# ---------------------------------------------------------------- C tokens
TOK = r"[A-Za-z_]\w*|\d[\w.]*|->|\+\+|--|<<=?|>>=?|[<>=!]=|&&|\|\||[-+*/%&|^]=|[^\s\w]"
KW = """auto break case char const continue default do double else enum extern float for goto if inline int long register
restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool size_t NULL true
false printf scanf malloc calloc realloc free strlen strcmp strcpy strcat memcpy memset sqrt pow abs fabs""".split()

def c_tokens(code):
    return (code.str.replace_all(r'"(?:\\.|[^"\\\n])*"', " S ").str.replace_all(r"'(?:\\.|[^'\\\n])*'", " C ")
            .str.replace_all(r"(?s)/\*.*?\*/", " ").str.replace_all(r"//[^\n]*", " ")
            .str.replace_all(r"(?m)^[ \t]*#[^\n]*", " ").str.extract_all(TOK))

canon = lambda tok: pl.when(tok.str.contains(r"^[A-Za-z_]") & ~tok.is_in(KW)).then(pl.lit("I")).otherwise(tok)

# ---------------------------------------------------------------- x86 instructions
DIRECTIVES = (r"(?m)^\s*\.(cfi_\w+|p2align|section|build_version|globl|subsections_via_symbols|private_extern"
              r"|weak_definition|weak_reference|alt_entry|loh|data_region|end_data_region|indirect_symbol|zerofill)[^\n]*\n?")

def asm_ins(asm):
    norm = (asm.str.replace_all(r"##[^\n]*", "").str.replace_all(DIRECTIVES, "")
            .str.replace_all(r"\bl?L[A-Za-z_.$]*\d*(_\d+)?(\.\d+)?\b", "L")
            .str.replace_all(r"\b_[A-Za-z_][\w.$]*", "_S").str.replace_all(r"[ \t]+", " "))
    return norm.str.split("\n").list.eval(pl.element().str.strip_chars()).list.eval(
        pl.element().filter(~pl.element().str.contains(r"^\.|:$|^$")))

def asm_fns(df):
    """df(probe, x86) -> (probe, fn, body) per `-- Begin function` block."""
    return (df.with_columns(part=pl.col("x86").str.split("-- Begin function ").list.slice(1)).explode("part")
            .select("probe", fn=pl.col("part").str.extract(r"^(\S+)"),
                    body=pl.col("part").str.extract(r"(?s)^[^\n]*\n(.*?)-- End function")))

# ---------------------------------------------------------------- probes
raw = f"{G}/data/raw"
he_id = pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1)
mbpp_src = pl.read_parquet(f"{raw}/decompile_eval_mbpp.parquet").sort("index").with_row_index("rank")
R = f"{S}/bringup_src/repo"
bu_x86 = pl.read_parquet(f"{S}/bench/bringup_O2.parquet").select(probe="bringup/" + pl.col("file").str.replace(r"\.s$", ""), x86="x86")
read = lambda f: open(f, errors="replace").read()
c_probes = pl.concat([
    pl.read_parquet(f"{raw}/asmbench_source.parquet").select(probe=he_id, code="code"),
    mbpp_src.select(probe=pl.format("mbpp/{}", "rank"), code=pl.concat_str("func_dep", "func", separator="\n")),
    pl.DataFrame([{"probe": p, "code": "\n".join(read(f) for f in sorted(glob.glob(f"{R}/{p[8:]}/*.c")))}
                  for p in bu_x86["probe"] if os.path.isdir(f"{R}/{p[8:]}")]),
]).with_row_index("id")
asm_probes = asm_fns(pl.concat([
    pl.read_parquet(f"{raw}/gg-bench-armv8-O2.parquet").select(probe=he_id, x86="x86"),
    pl.read_parquet(f"{G}/data/mbpp_armv8_O2.parquet").select(probe="task_id", x86="x86"),
    bu_x86,
])).with_row_index("id")

def probe_table(df, k, transform=None):
    p = shingles(df, k, transform)
    generic = p.group_by("h").len().filter(pl.col("len") >= MAX_PROBES)["h"]
    return p.filter(~pl.col("h").is_in(generic.implode()))

PC = probe_table(c_probes.select("id", tok=c_tokens(pl.col("code"))), K_C, canon)
PA = probe_table(asm_probes.select("id", tok=asm_ins(pl.col("body"))), K_ASM)
log("C probes", c_probes.height, "shingles", PC.height, "| asm probe fns", asm_probes.height, "shingles", PA.height)

# ---------------------------------------------------------------- corpus scans
def scan(batches, tok, k, P, transform=None):
    """batches yield DataFrame(file, text); returns hits(doc, h) and the doc->file table."""
    keep, hits, files, base = P["h"].unique().implode(), [], [], 0
    for b in batches:
        b = b.with_row_index("id", offset=base)
        base += b.height
        files.append(b.select("id", "file"))
        hits.append(shingles(b.select("id", tok=tok(pl.col("text"))), k, transform, keep))
        log(f"  {base} docs, {sum(h.height for h in hits)} hits")
    return pl.concat(hits).rename({"id": "doc"}), pl.concat(files).rename({"id": "doc"})

def stack_batches():
    sft_stack = pl.read_parquet(f"{S}/sft_paths.parquet").filter(pl.col("source") == "stack").select("file")
    n = 0
    for f in sorted(glob.glob(f"{S}/stackc/*.parquet")):
        d = pl.read_parquet(f).join(sft_stack, on="file", how="semi").rename({"data": "text"})
        for b in d.iter_slices(BATCH):
            if LIMIT and n >= LIMIT:
                return
            n += b.height
            yield b

def sft_batches():
    n = 0
    for f in sorted(glob.glob(f"{S}/sft/train-*.parquet")):
        for b in pl.read_parquet(f, columns=["file", "x86"]).rename({"x86": "text"}).iter_slices(BATCH):
            if LIMIT and n >= LIMIT:
                return
            n += b.height
            yield b

def score(hits, files, P, probes):
    common = hits.group_by("h").len().filter(pl.col("len") > MAX_DF)["h"]
    P = P.filter(~pl.col("h").is_in(common.implode()))
    size = P.group_by("id").len("size")
    return (hits.join(P, on="h").group_by("doc", "id").len("shared")
            .join(size, on="id").with_columns(containment=pl.col("shared") / pl.col("size"))
            .sort("containment", "shared", descending=True).group_by("id", maintain_order=True).head(3)
            .join(files, on="doc").join(probes.drop("code", "body", strict=False), on="id")
            .drop("doc", "id").sort("containment", descending=True)), size.join(probes.drop("code", "body", strict=False), on="id")

log("C level: SFT Stack files")
c_res, c_size = score(*scan(stack_batches(), c_tokens, K_C, PC, canon), PC, c_probes)
c_res.write_parquet(f"{S}/neardup_c.parquet"); c_size.write_parquet(f"{S}/neardup_c_size.parquet")
log("x86 level: all SFT rows")
a_res, a_size = score(*scan(sft_batches(), asm_ins, K_ASM, PA), PA, asm_probes)
a_res.write_parquet(f"{S}/neardup_asm.parquet"); a_size.write_parquet(f"{S}/neardup_asm_size.parquet")
log("done")
