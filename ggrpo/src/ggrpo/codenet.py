"""CodeNet pool (Mac): CodeNet's accepted C solutions of AOJ and AtCoder problems x the sample cases of each statement
(CDLA-Permissive-2.0, no scraping). Writes two tables:
- problems: one row per problem with its judge, its samples (stripped, in statement order) and its split;
- solutions: one row per distinct accepted solution (whitespace-insensitive, lowest submission id kept).
Problems with fewer than codenet.min_outputs distinct sample outputs are SFT-only. The others are split by group
(problems with identical samples are one group: AtCoder reuses problems across contests): sha256(seed:group) alone
places each group, so the fractions are approximate, and the test split depends neither on library versions nor on
which other problems are eligible.
No asm here: the VM build comes next."""
import hashlib
import itertools
import re

import lxml.html
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
# "Sample Input 1" / "Output for (the) Sample Input 1", 入力例 1 / 出力例 1, サンプル入力 1, 入力1 / 出力1
sample_input = re.compile(r"sample input|入力例|サンプル入力|^入力\s*\d+$", re.IGNORECASE)
sample_output = re.compile(r"sample output|output for (?:the )?sample|出力例|サンプル出力|^出力\s*\d+$", re.IGNORECASE)
sample = pl.Struct({"input": pl.String, "output": pl.String})


def parse_samples(statement: str) -> list[dict[str, str]] | None:
    """Each <pre> right after a sample heading (AOJ; AtCoder in Japanese and English, so pairs are deduplicated).
    None when the inputs and outputs do not pair up: that problem is left out of RL, and counted in the log."""
    inputs, outputs, wanted = [], [], None
    for element in lxml.html.fragment_fromstring(statement, create_parent="div").iter(
        "h1", "h2", "h3", "h4", "h5", "h6", "pre"
    ):
        if element.tag != "pre":
            heading = element.text_content().strip()
            wanted = outputs if sample_output.search(heading) else inputs if sample_input.search(heading) else None
        elif wanted is not None:
            wanted.append(element.text_content().strip())
            wanted = None
    if len(inputs) != len(outputs):
        return None
    return [{"input": i, "output": o} for i, o in dict.fromkeys(zip(inputs, outputs))]


fractions = settings.codenet.split
if abs(sum(fractions.values()) - 1) > 1e-9:
    raise ValueError(f"codenet.split must sum to 1: {fractions}")

submissions = (
    pl.scan_parquet(settings.codenet.submissions.download() / "C")
    .filter(status="Accepted")
    .select(problem_id="p_id", solution_id="s_id", aoj=pl.col("original_language") == settings.codenet.aoj_language,
            c="code")
    .collect()
)
judges = submissions.group_by("problem_id").agg(pl.col("aoj").n_unique())
if (judges["aoj"] > 1).any():
    raise ValueError(f"problems with both AOJ and AtCoder submissions: {judges.filter(pl.col('aoj') > 1)}")
solutions = (
    submissions.sort("problem_id", "solution_id")
    .with_columns(key=pl.col("c").str.replace_all(r"\s+", ""))
    .unique(["problem_id", "key"], keep="first", maintain_order=True)
    .drop("key")
)

metadata = pl.read_parquet(settings.codenet.metadata.download() / "data").select(
    problem_id="id", title=pl.col("slug_name").str.strip_chars(), statement="pretty_content"
)
problems = (
    solutions.group_by("problem_id")
    .agg(judge=pl.when(pl.col("aoj").first()).then(pl.lit("AOJ")).otherwise(pl.lit("AtCoder")), solutions=pl.len())
    .join(metadata, on="problem_id", how="left", validate="1:1")
)
if problems["statement"].is_null().any():
    raise ValueError(f"problems without metadata: {problems.filter(pl.col('statement').is_null())['problem_id']}")
problems = problems.with_columns(
    samples=pl.col("statement").map_elements(parse_samples, return_dtype=pl.List(sample))
).with_columns(
    eligible=(pl.col("samples").list.eval(pl.element().struct.field("output")).list.n_unique()
              >= settings.codenet.min_outputs).fill_null(False),   # unpaired samples: not an RL problem
    # AtCoder reuses a problem across concurrent contests (p03945 = p03947): identical samples make one group, which
    # lands in one split; it is named by its smallest problem id
    group=pl.col("problem_id").min().over(
        pl.when(pl.col("samples").list.len() > 0)
        .then(pl.col("samples").list.eval(pl.concat_str(pl.element().struct.field("input"), pl.lit("\x1f"),
                                                        pl.element().struct.field("output"))).list.sort().list.join("\x1e"))
        .otherwise(pl.col("problem_id"))
    ),
)

rl = (
    problems.filter("eligible")
    .with_columns(quantile=pl.col("group").map_elements(   # uniform in [0, 1), from this group alone
        lambda group: int(hashlib.sha256(f"{settings.codenet.seed}:{group}".encode()).hexdigest()[:15], 16) / 16**15,
        return_dtype=pl.Float64,
    ))
    .select("problem_id", split=pl.coalesce(   # the first split whose cumulative fraction exceeds the quantile
        pl.when(pl.col("quantile") < bound).then(pl.lit(name))
        for name, bound in zip(fractions, itertools.accumulate(fractions.values()))
    ))
)
if rl["split"].is_null().any():
    raise ValueError(f"problems past the last cumulative fraction: {rl.filter(pl.col('split').is_null())}")
problems = (
    problems.join(rl, on="problem_id", how="left", validate="1:1")
    .with_columns(pl.col("split").fill_null("sft_only"))
    .select("problem_id", "group", "judge", "title", "split", "samples", "solutions")
    .sort("problem_id")
)
solutions = solutions.join(problems.select("problem_id", "judge", "split"), on="problem_id", validate="m:1").select(
    "problem_id", "solution_id", "judge", "split", "c"
)

settings.codenet.problems.parent.mkdir(parents=True, exist_ok=True)
problems.write_parquet(settings.codenet.problems)
solutions.write_parquet(settings.codenet.solutions)
counts = problems.group_by("judge", "split").agg(problems=pl.len(), solutions=pl.col("solutions").sum()).sort(
    "judge", "split"
)
log.info("wrote", problems=str(settings.codenet.problems), solutions=str(settings.codenet.solutions),
         submissions=submissions.height, distinct=solutions.height,
         grouped=problems.filter(pl.col("group").is_duplicated())["problem_id"].len(),
         unpaired_samples=problems["samples"].is_null().sum(), counts=counts.to_dicts())
