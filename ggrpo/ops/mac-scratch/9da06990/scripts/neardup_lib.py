import functools, operator
import polars as pl
MULT = [pl.lit((0x9E3779B97F4A7C15 * (2 * i + 1)) % 2**64 | 1, dtype=pl.UInt64) for i in range(16)]
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

