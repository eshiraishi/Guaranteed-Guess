"""Planted copies for ggrpo.decontam (Mac, --extra data): C of programs that are not eval items, copied with known
edits (symbols renamed, constants changed, statements inserted or deleted at a set rate, int widened, other -O levels).
Writes decontam.canary_c; ggrpo.canary_build compiles it in the VM. The run must find these copies at
decontam.canary_recall."""
import hashlib
import random

import polars as pl
import structlog
from pydantic_settings import CliApp
from pygments.lexers import CLexer
from pygments.token import Keyword, Literal, Name, Text

from ggrpo.decontam import read_mbpp_c
from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
lexer = CLexer()
RATES = (0.05, 0.1, 0.2, 0.4)


def find_next_code(tokens: list, i: int) -> int | None:
    return next((j for j in range(i + 1, len(tokens)) if tokens[j][1].strip()), None)


def rename_symbols(tokens: list) -> list:
    """Renames the file's defined functions (except main) and its global variables."""
    names, brace, paren = set(), 0, 0
    for i, (kind, value) in enumerate(tokens):
        brace += (value == "{") - (value == "}")
        paren += (value == "(") - (value == ")")
        following = find_next_code(tokens, i)
        if following is None or kind not in Name:
            continue
        if kind in Name.Function and value != "main":
            depth, closing = 0, following
            while closing < len(tokens):
                depth += (tokens[closing][1] == "(") - (tokens[closing][1] == ")")
                if depth == 0:
                    break
                closing += 1
            body = find_next_code(tokens, closing) if closing < len(tokens) else None
            if body is not None and tokens[body][1] == "{":
                names.add(value)
        elif brace == 0 and paren == 0 and tokens[following][1] in ("=", "[", ";", ","):
            names.add(value)
    return [(kind, f"{value}_zq" if kind in Name and value in names else value) for kind, value in tokens]


def change_constants(tokens: list, rng: random.Random, rate: float) -> list:
    return [(kind, str(int(value) + 1) if kind in Literal.Number.Integer and value.isdigit() and rng.random() < rate
             else value) for kind, value in tokens]


def find_statement_ends(tokens: list) -> list[int]:
    """Indices of the semicolons that end a statement inside a function body."""
    found, brace, paren = [], 0, 0
    for i, (_, value) in enumerate(tokens):
        brace += (value == "{") - (value == "}")
        paren += (value == "(") - (value == ")")
        if value == ";" and paren == 0 and brace >= 1:
            found.append(i)
    return found


def insert_statements(tokens: list, rng: random.Random, rate: float) -> list:
    chosen = {i for i in find_statement_ends(tokens) if rng.random() < rate}
    edited = [(Text, "static volatile int ggv;\n")]
    for i, token in enumerate(tokens):
        edited.append(token)
        if i in chosen:
            edited.append((Text, " ggv++;"))
    return edited


def delete_statements(tokens: list, rng: random.Random, rate: float) -> list:
    removed: set[int] = set()
    previous = -1
    for end in find_statement_ends(tokens):
        start = max((j for j in range(previous + 1, end) if tokens[j][1] in ("{", "}")), default=previous) + 1
        first = find_next_code(tokens, start - 1)
        previous = end
        if (first is None or first >= end or tokens[first][0] in Keyword
                or any(tokens[j][1] in ("{", "}") for j in range(first, end))):
            continue
        if rng.random() < rate:
            removed.update(range(first, end + 1))
    return [token for i, token in enumerate(tokens) if i not in removed]


def widen_types(tokens: list) -> list:
    """int becomes long, except main's return type."""
    edited = []
    for i, (kind, value) in enumerate(tokens):
        following = find_next_code(tokens, i)
        is_main_type = following is not None and tokens[following][1] == "main"
        edited.append((kind, "long" if value == "int" and kind in Keyword and not is_main_type else value))
    return edited


def build_variants(code: str, seed: str) -> dict[str, tuple[str, str]]:
    """variant -> (C, optimization flag)."""
    tokens = list(lexer.get_tokens(code))
    make_rng = lambda name: random.Random(hashlib.sha256(f"{seed}/{name}".encode()).digest())
    join_text = lambda edited: "".join(value for _, value in edited)
    variants = {"exact": (code, "-O2"), "rename": (join_text(rename_symbols(tokens)), "-O2"),
                "retype": (join_text(widen_types(tokens)), "-O2"), "flag_O1": (code, "-O1"), "flag_O3": (code, "-O3"),
                "flag_Os": (code, "-Os")}
    for rate in RATES:
        variants[f"const_{rate}"] = (join_text(change_constants(tokens, make_rng(f"c{rate}"), rate)), "-O2")
        variants[f"insert_{rate}"] = (join_text(insert_statements(tokens, make_rng(f"i{rate}"), rate)), "-O2")
        variants[f"delete_{rate}"] = (join_text(delete_statements(tokens, make_rng(f"d{rate}"), rate)), "-O2")
    return variants


programs, seed = settings.decontam.canary_programs, settings.decontam.canary_seed
contest = pl.read_parquet(settings.codenet.solutions).filter(pl.col("split").is_in(["sft_only", "train"])).sample(
    programs, seed=seed, shuffle=True).select(base=pl.format("contest/{}", "solution_id"), c="c")
short = read_mbpp_c().filter(pl.col("item_id").is_in(pl.read_ndjson(settings.data.train)["task_id"].implode())).sample(
    programs, seed=seed, shuffle=True).select(base=pl.format("short/{}", "item_id"), c="c")
stack = pl.concat([pl.scan_parquet(path).select("file", "data") for dataset in settings.decontam.stack
                   for path in sorted(dataset.download().rglob("*.parquet"))]).collect()
repo = stack.sample(programs, seed=seed, shuffle=True).select(base=pl.format("repo/{}", "file"), c="data")
rows = [{"base": base, "variant": variant, "flag": flag, "c": edited}
        for base, code in pl.concat([contest, short, repo]).iter_rows()
        for variant, (edited, flag) in build_variants(code, f"{seed}/{base}").items()]
settings.decontam.canary_c.parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).write_parquet(settings.decontam.canary_c)
log.info("wrote", path=str(settings.decontam.canary_c), variants=len(rows), programs=3 * programs)
