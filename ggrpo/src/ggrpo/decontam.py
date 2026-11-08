"""Decontamination (Mac, --extra data): which training documents leak an eval item.
Eval items: the tasks of every file in decontam.evals (MBPP-C dev, HumanEval-C, CodeNet val and test) and BringUp.
Training documents: GG's O2 SFT pairs, the SFT side of CodeNet (SFT-only and train problems) and MBPP-C train.

Writes (document, item, rule, score, drop) for every score at or above decontam.report; downstream, a document with any
drop row is left out. The rules:
- asm: x86 split at Darwin symbol labels and lexed with pygments' GAS lexer, without comments or directives, local
  labels as "L". An eval function counts with at least decontam.min_instructions instructions and decontam.min_rare rare
  n-grams. Score: the largest share of an item function's rare n-grams found in one training function. Drops at
  decontam.drop_asm.
- exact: every item function of at least decontam.min_exact instructions is token-exact in the document. For short
  items the asm rule cannot check. Always drops.
- c: canonical C tokens (identifiers masked). The item needs decontam.min_rare_c rare n-grams. Score: the share of them
  in the document. Drops at decontam.drop_c. AnghaBench documents are single functions, so this rule sees one only when
  it holds that share of the whole item.
An n-gram is rare when at most decontam.max_df of the pooled samples (decontam.df_sample documents per source) has it.

Also writes, per eval item, which rules could check it (decontam.items): an item no rule could check is not clean, only
unmeasured.

Checked during the run, which fails if they do not hold: planted copies (ggrpo.canary) are found at
decontam.canary_recall, and each eval item with a function of min_exact instructions finds itself. Checks are not
written to the output."""
import hashlib
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import polars as pl
import structlog
from pydantic_settings import CliApp
from pygments.lexers import GasLexer
from pygments.token import Comment

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
lexer = GasLexer()
symbol = re.compile(r"^(_[\w$.]+):")   # Darwin functions and data objects; local labels start with L or l
label = re.compile(r"^[\w$.]+:")
local_label = re.compile(r"^(?:LBB|LCPI|LJTI|Ltmp|Lloh|Lfunc|Lexception|LCFI|L_|l_)[\w.$]*:?$")
x86 = pl.col("prompt").list.first().struct.field("content").str.extract(r"(?s)```x86asm\n(.*)\n```$", 1)
c_token = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|->|\+\+|--|<<=?|>>=?|[<>=!]=|&&|\|\||[-+*/%&|^]=|[^\s\w]")
c_string = re.compile(r'"(?:\\.|[^"\\\n])*"')
c_char = re.compile(r"'(?:\\.|[^'\\\n])*'")
c_comment = re.compile(r"(?s)/\*.*?\*/|//[^\n]*")
c_directive = re.compile(r"(?m)^[ \t]*#[^\n]*")
c_keywords = frozenset([   # the data survey's list: these stay visible, every other identifier becomes I
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float",
    "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "bool", "size_t", "NULL",
    "true", "false", "printf", "scanf", "malloc", "calloc", "realloc", "free", "strlen", "strcmp", "strcpy", "strcat",
    "memcpy", "memset", "sqrt", "pow", "abs", "fabs",
])


def split_functions(asm: str) -> dict[str, tuple[int, list[str]]]:
    """(instruction count, canonical tokens) of every symbol with instructions; data objects have none."""
    lines: dict[str, list[str]] = {}
    name = None
    for line in asm.splitlines():
        code = re.sub(r"(?:##|;).*$", "", line).strip()   # Apple's x86 and arm comments; a leading \s* is quadratic
        if match := symbol.match(code):
            name = match.group(1)
            lines[name] = []
        elif name is not None and code and not code.startswith("."):
            lines[name].append(code)
    functions = {}
    for name, code in lines.items():
        instructions = sum(1 for line in code if not label.match(line))
        if instructions:
            tokens = [("L" if local_label.match(value) else value)
                      for kind, value in lexer.get_tokens("\n".join(code)) if kind not in Comment and value.strip()]
            functions[name] = (instructions, tokens)
    return functions


def tokenize_c(code: str) -> list[str]:
    """The data survey's canonical C tokens: no comments or directives, literals and identifiers masked."""
    code = c_directive.sub(" ", c_comment.sub(" ", c_char.sub(" C ", c_string.sub(" S ", code))))
    return ["I" if (token[0].isalpha() or token[0] == "_") and token not in c_keywords else token
            for token in c_token.findall(code)]


def hash_tokens(tokens: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b("\x1f".join(tokens).encode(), digest_size=8).digest())


def hash_ngrams(tokens: list[str]) -> set[int]:
    n = settings.decontam.n
    return {hash_tokens(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def count_frequency(documents: list[str], ngrams: frozenset[int]) -> tuple[Counter, int]:
    """How many training functions contain each eval n-gram, and how many training functions there were."""
    counts: Counter = Counter()
    functions = 0
    for asm in documents:
        for _, tokens in split_functions(asm).values():
            counts.update(hash_ngrams(tokens) & ngrams)
            functions += 1
    return counts, functions


def count_c_frequency(documents: list[str], ngrams: frozenset[int]) -> tuple[Counter, int]:
    """How many training documents contain each eval C n-gram, and how many documents there were."""
    counts: Counter = Counter()
    for code in documents:
        counts.update(hash_ngrams(tokenize_c(code)) & ngrams)
    return counts, len(documents)


def find_overlaps(documents: list[tuple[str, str]], index: dict[int, list[str]], sizes: dict[str, int],
                  required: dict[int, list[str]], needed: dict[str, int]) -> list[tuple[str, str, str, float]]:
    """(document, eval item, rule, score) of the asm rule at or above decontam.report, and of the exact rule."""
    rows = []
    for document_id, asm in documents:
        best: dict[str, float] = {}
        digests = set()
        for _, tokens in split_functions(asm).values():
            digests.add(hash_tokens(tokens))
            shared = Counter(function for ngram in hash_ngrams(tokens) & index.keys() for function in index[ngram])
            for function, count in shared.items():
                item = function.split("::")[0]
                best[item] = max(best.get(item, 0.0), count / sizes[function])
        rows += [(document_id, item, "asm", score) for item, score in best.items() if score >= settings.decontam.report]
        matched = Counter(item for digest in digests & required.keys() for item in required[digest])
        rows += [(document_id, item, "exact", 1.0) for item, count in matched.items() if count == needed[item]]
    return rows


def find_c_overlaps(documents: list[tuple[str, str]], index: dict[int, list[str]],
                    sizes: dict[str, int]) -> list[tuple[str, str, str, float]]:
    """(document, eval item, "c", score) at or above decontam.report."""
    rows = []
    for document_id, code in documents:
        shared = Counter(item for ngram in hash_ngrams(tokenize_c(code)) & index.keys() for item in index[ngram])
        rows += [(document_id, item, "c", count / sizes[item]) for item, count in shared.items()
                 if count / sizes[item] >= settings.decontam.report]
    return rows


def split_batches(documents: list, size: int) -> list[list]:
    return [documents[i:i + size] for i in range(0, len(documents), size)]


def read_mbpp_c() -> pl.DataFrame:
    """(item_id, c) of every MBPP-C task, with the ids ggrpo.build gives them."""
    return pl.read_parquet(settings.build.source).sort("index").with_row_index("rank").select(
        item_id=pl.format("mbpp/{}", "rank"), c=pl.concat_str("func_dep", "func", separator="\n")
    )


def read_evals() -> pl.DataFrame:
    """Eval items as (item_id, set, x86, c)."""
    tasks = pl.concat([
        *(pl.read_ndjson(path).select(item_id="task_id", set=pl.lit(path.stem), x86=x86)
          for path in settings.decontam.evals),
        pl.read_parquet(settings.decontam.bringup.download() / "data").select(
            item_id=pl.format("bringup/{}", pl.col("file").str.strip_suffix(".s")), set=pl.lit("bringup"), x86="x86"
        ),
    ])
    humaneval = pl.read_parquet(settings.data.humaneval_tests).select(
        item_id=pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1), c="code"
    )
    codenet = pl.read_parquet(settings.codenet.solutions).select(
        item_id=pl.format("codenet/{}/{}", "problem_id", "solution_id"), c="c"
    )
    bringup = pl.DataFrame(
        [(f"bringup/{folder.name}", "\n".join(path.read_text(errors="replace") for path in sorted(folder.glob("*.c"))))
         for folder in settings.decontam.bringup_c.iterdir() if any(folder.glob("*.c"))],
        schema={"item_id": pl.String, "c": pl.String}, orient="row",
    )
    return tasks.join(pl.concat([humaneval, read_mbpp_c(), codenet, bringup]), on="item_id", how="left", validate="1:1")


def read_canaries() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Planted programs as eval items (item_id, set, x86, c) and their copies as documents (document_id, x86, c).
    Programs without a function of min_exact instructions are left out: no real eval item has that shape."""
    planted = pl.read_parquet(settings.decontam.canary_c).join(
        pl.read_parquet(settings.decontam.canary_x86), on=["base", "variant"], validate="1:1"
    ).filter(pl.col("x86").is_not_null())
    originals = planted.filter(pl.col("variant") == "exact")
    usable = [base for base, asm in originals.select("base", "x86").iter_rows()
              if any(count >= settings.decontam.min_exact for count, _ in split_functions(asm).values())]
    planted = planted.filter(pl.col("base").is_in(usable))
    originals = planted.filter(pl.col("variant") == "exact")
    copies = planted.join(originals.select("base", original="x86"), on="base").filter(
        (pl.col("variant") == "exact") | (pl.col("x86") != pl.col("original"))
    )
    return (originals.select(item_id=pl.format("canary/{}", "base"), set=pl.lit("canary"), x86="x86", c="c"),
            copies.select(document_id=pl.format("canary/{}/{}", "base", "variant"), x86="x86", c="c"))


def select_fraction(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Keeps decontam.fraction of the documents, for a smoke test; 1.0 in a real run."""
    return frame.filter(pl.col("document_id").hash(seed=0).cast(pl.Float64) / 2**64 <= settings.decontam.fraction)


def read_codenet() -> pl.LazyFrame:
    """The SFT side of CodeNet with its x86 as (document_id, x86, c)."""
    solutions = pl.read_parquet(settings.codenet.solutions).filter(pl.col("split").is_in(["sft_only", "train"]))
    built = pl.read_parquet(settings.codenet.built / "part-*.parquet")
    if missing := solutions.join(built, on="solution_id", how="anti").height:
        raise ValueError(f"{missing} SFT-side CodeNet solutions are not in {settings.codenet.built} yet")
    return solutions.join(built, on="solution_id", validate="1:1").filter(pl.col("x86").is_not_null()).select(
        document_id=pl.format("codenet/{}/{}", "problem_id", "solution_id"), x86="x86", c="c"
    ).lazy()


def read_documents() -> dict[str, list[pl.LazyFrame]]:
    """Training documents as (document_id, x86) per source: one frame per GG file, CodeNet and MBPP-C."""
    gg = [pl.scan_parquet(path).select(document_id="file", x86="x86")
          for path in sorted(settings.decontam.gg.download().rglob("*.parquet"))]
    mbpp = pl.read_ndjson(settings.data.train).select(document_id="task_id", x86=x86)
    return {"gg": [select_fraction(frame) for frame in gg],
            "codenet": [select_fraction(read_codenet().select("document_id", "x86"))],
            "mbpp": [select_fraction(mbpp.lazy())]}


def read_c_documents() -> dict[str, list[pl.LazyFrame]]:
    """Training documents as (document_id, c) per source. Every GG document must have its C."""
    gg_files = pl.scan_parquet(str(settings.decontam.gg.download() / "data" / "*.parquet")).select("file").collect()
    stack = pl.concat([pl.scan_parquet(path).select("file", c="data") for dataset in settings.decontam.stack
                       for path in sorted(dataset.download().rglob("*.parquet"))]).collect()
    angha = pl.read_parquet(settings.decontam.angha_c)
    if stack["file"].is_duplicated().any():
        raise ValueError("the Stack C datasets repeat a file")
    if missing := gg_files.join(pl.concat([stack, angha]), on="file", how="anti").height:
        raise ValueError(f"{missing} GG documents have no C")
    mbpp = pl.read_ndjson(settings.data.train).select(item_id="task_id").join(read_mbpp_c(), on="item_id",
                                                                              validate="1:1")
    return {source: [select_fraction(frame.join(gg_files, on="file", how="semi")
                                     .select(document_id="file", c="c").lazy())]
            for source, frame in (("stack", stack), ("angha", angha))} | {
        "codenet": [select_fraction(read_codenet().select("document_id", "c"))],
        "mbpp": [select_fraction(mbpp.select(document_id="item_id", c="c").lazy())],
    }


def find_common(pool: ProcessPoolExecutor, sources: dict[str, list[pl.LazyFrame]], column: str, count,
                universe: frozenset[int]) -> set[int]:
    """n-grams of the universe above decontam.max_df in the pooled samples of df_sample documents per source."""
    frequency: Counter = Counter()
    units = 0
    for source, frames in sources.items():
        rows = sum(frame.select(pl.len()).collect().item() for frame in frames)
        fraction = min(1.0, settings.decontam.df_sample / rows)
        sample = [text for frame in frames
                  for text in frame.select(column).collect().sample(fraction=fraction, seed=0)[column]]
        for counts, seen in pool.map(partial(count, ngrams=universe), split_batches(sample, settings.decontam.batch)):
            frequency.update(counts)
            units += seen
        log.info("frequency", source=source, column=column, documents=len(sample), units=units)
    return {ngram for ngram, value in frequency.items() if value > settings.decontam.max_df * units}


def scan_frames(pool: ProcessPoolExecutor, sources: dict[str, list[pl.LazyFrame]],
                scan) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    for source, frames in sources.items():
        for frame in frames:
            for found in pool.map(scan, split_batches(frame.collect().rows(), settings.decontam.batch)):
                rows += found
        log.info("scanned", source=source, pairs=len(rows))
    return rows


def check_self(pairs: pl.DataFrame, items: list[str]) -> None:
    """Every eval item with a function of min_exact instructions must flag its own copy."""
    found = set(pairs.filter(pl.col("rule") == "exact", pl.col("document_id").str.starts_with("self/"),
                             pl.col("document_id") == pl.lit("self/") + pl.col("item_id"))["item_id"])
    if missing := sorted(set(items) - found):
        raise ValueError(f"{len(missing)} eval items did not find themselves: {missing[:10]}")


def check_canaries(pairs: pl.DataFrame, copies: pl.DataFrame) -> None:
    """Planted copies must be dropped at decontam.canary_recall, per program type and kind of copy."""
    flagged = pairs.filter("drop", pl.col("item_id").str.starts_with("canary/"),
                           pl.col("document_id").str.contains(r"^canary/")).with_columns(
        base=pl.col("document_id").str.extract(r"^(canary/.*)/[^/]+$", 1)
    ).filter(pl.col("base") == pl.col("item_id")).select("document_id").unique().with_columns(hit=pl.lit(True))
    recall = copies.select("document_id").join(flagged, on="document_id", how="left").with_columns(
        domain=pl.col("document_id").str.extract(r"^canary/([^/]+)/", 1),
        kind=pl.col("document_id").str.extract(r"/([^/]+)$", 1),
        hit=pl.col("hit").fill_null(False),
    ).with_columns(
        group=pl.when(pl.col("kind").str.contains(r"^(?:const|insert|delete)_"))
        .then(pl.lit("edit_") + pl.col("kind").str.split("_").list.last()).otherwise("kind")
    ).group_by("domain", "group").agg(recall=pl.col("hit").mean(), copies=pl.len()).sort("domain", "group")
    log.info("canaries", recall=recall.to_dicts())
    required = pl.DataFrame(list(settings.decontam.canary_recall.items()), schema=["group", "required"], orient="row")
    failed = recall.join(required, on="group").filter(pl.col("recall") < pl.col("required"))
    if failed.height:
        raise ValueError(f"canary recall below the protocol's bar: {failed.to_dicts()}")


if __name__ == "__main__":   # pool workers import this module
    real = read_evals()
    canary_items, canary_documents = read_canaries()
    evals = pl.concat([real, canary_items])
    if evals["x86"].is_null().any() or evals["c"].is_null().any() or evals["item_id"].is_duplicated().any():
        raise ValueError("eval items need a unique id, an x86 prompt and C")
    profiles = {item: split_functions(asm) for item, asm in evals.select("item_id", "x86").iter_rows()}
    candidates = {   # "item_id::function" -> n-grams, for eval functions with enough instructions
        f"{item}::{name}": hash_ngrams(tokens)
        for item, functions in profiles.items()
        for name, (instructions, tokens) in functions.items()
        if instructions >= settings.decontam.min_instructions
    }
    required: dict[int, list[str]] = {}    # hash of a whole function -> items that need it for the exact rule
    needed: Counter = Counter()
    for item, functions in profiles.items():
        for digest in {hash_tokens(tokens) for count, tokens in functions.values()
                       if count >= settings.decontam.min_exact}:
            required.setdefault(digest, []).append(item)
            needed[item] += 1
    c_grams = {item: hash_ngrams(tokenize_c(code)) for item, code in evals.select("item_id", "c").iter_rows()}
    selfs = real.select(document_id=pl.format("self/{}", "item_id"), x86="x86", c="c").lazy()
    documents = read_documents()
    c_documents = read_c_documents()
    checks = {"canary": canary_documents.lazy(), "self": selfs}
    with ProcessPoolExecutor(settings.decontam.workers) as pool:
        common = find_common(pool, {source: documents[source] for source in ("gg", "codenet")}, "x86", count_frequency,
                             frozenset().union(*candidates.values()))
        rare = {function: grams - common for function, grams in candidates.items()}
        sizes = {function: len(grams) for function, grams in rare.items() if len(grams) >= settings.decontam.min_rare}
        index: dict[int, list[str]] = {}
        for function in sizes:
            for ngram in rare[function]:
                index.setdefault(ngram, []).append(function)
        log.info("asm evals", items=evals.height, functions=len(candidates), checked=len(sizes),
                 items_checked=len({function.split("::")[0] for function in sizes}), common_ngrams=len(common),
                 index=len(index))
        scan = partial(find_overlaps, index=index, sizes=sizes, required=required, needed=dict(needed))
        rows = scan_frames(pool, documents | {name: [frame.select("document_id", "x86")]
                                              for name, frame in checks.items()}, scan)

        common_c = find_common(pool, {source: c_documents[source] for source in ("stack", "angha", "codenet")}, "c",
                               count_c_frequency, frozenset().union(*c_grams.values()))
        rare_c = {item: grams - common_c for item, grams in c_grams.items()}
        sizes_c = {item: len(grams) for item, grams in rare_c.items() if len(grams) >= settings.decontam.min_rare_c}
        index_c: dict[int, list[str]] = {}
        for item in sizes_c:
            for ngram in rare_c[item]:
                index_c.setdefault(ngram, []).append(item)
        log.info("c evals", items=len(c_grams), checked=len(sizes_c), common_ngrams=len(common_c), index=len(index_c))
        scan_c = partial(find_c_overlaps, index=index_c, sizes=sizes_c)
        rows += scan_frames(pool, c_documents | {name: [frame.select("document_id", "c")]
                                                 for name, frame in checks.items()}, scan_c)
    drop = (pl.col("rule") == "exact") | ((pl.col("rule") == "asm") & (pl.col("score") >= settings.decontam.drop_asm)) \
        | ((pl.col("rule") == "c") & (pl.col("score") >= settings.decontam.drop_c))
    schema = {"document_id": pl.String, "item_id": pl.String, "rule": pl.String, "score": pl.Float64}
    pairs = pl.DataFrame(rows, schema=schema, orient="row").with_columns(drop=drop)
    check_self(pairs, [item for item in real["item_id"] if needed[item]])
    check_canaries(pairs, canary_documents)
    output = pairs.filter(~pl.col("document_id").str.contains(r"^(?:self|canary)/"),
                          ~pl.col("item_id").str.starts_with("canary/"))
    settings.decontam.out.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(settings.decontam.out)
    asm_items = {function.split("::")[0] for function in sizes}
    real.select("item_id", "set").with_columns(
        asm_checked=pl.Series([item in asm_items for item in real["item_id"]]),
        c_checked=pl.Series([item in sizes_c for item in real["item_id"]]),
        exact_checked=pl.Series([needed[item] > 0 for item in real["item_id"]]),
    ).write_parquet(settings.decontam.items)
    source = pl.when(pl.col("document_id").str.starts_with("codenet/")).then(pl.lit("codenet")) \
        .when(pl.col("document_id").str.starts_with("mbpp/")).then(pl.lit("mbpp")).otherwise(pl.lit("gg"))
    dropped = output.filter("drop").with_columns(source=source)
    shares = {name: dropped.filter(pl.col("source") == name)["document_id"].n_unique()
              / sum(frame.select(pl.len()).collect().item() for frame in documents[name])
              for name in settings.decontam.budget}
    flagged_items = dropped.group_by("item_id").agg(documents=pl.col("document_id").n_unique())
    summary = real.select("item_id", "set").join(flagged_items, on="item_id", how="left").group_by("set").agg(
        items=pl.len(), items_flagged=pl.col("documents").is_not_null().sum(), documents=pl.col("documents").sum())
    log.info("wrote", path=str(settings.decontam.out), pairs=output.height, summary=summary.sort("set").to_dicts(),
             by_rule=dropped.group_by("rule", "source").agg(pl.col("document_id").n_unique()).sort("rule", "source")
             .to_dicts(), shares=shares)
    if over := {name: share for name, share in shares.items() if share > settings.decontam.budget[name]}:
        raise ValueError(f"over the protocol's budget, stop and ask: {over}")
