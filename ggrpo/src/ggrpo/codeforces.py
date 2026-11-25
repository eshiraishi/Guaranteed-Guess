"""Codeforces eval pool (Mac, --extra data): problems with complete official tests (open-r1/codeforces) x their
accepted GNU C solutions (open-r1/codeforces-submissions), one row per problem. Evaluation only (Enzo, Nov 22): the
tests never enter training or a reward and are never redistributed.

Kept: stdio; no generated checker (an LLM-written checker is a synthetic oracle); not interactive; complete official
tests (open-r1's flag: no test truncated; partial tests miss the long tests Codeforces truncates, so they are not
ground truth) with >= 2 distinct outputs (a program that prints a constant must not pass); >= 1 statement example (the
sample tests); an accepted solution. Per problem, up to codeforces.solutions distinct solutions (whitespace-insensitive,
lowest submission id kept) in sha256(seed:submission id) order, the order in which ggrpo.codeforces_tasks picks the
reference. Problems are split by group (identical examples make one group) from sha256(seed:group) alone, as in
ggrpo.codenet, so the split depends neither on the build nor on which other problems survive it.

Problem ids drop open-r1's slash ("755/D" -> "755D"): task ids are codeforces/<problem>/<submission>, and
ggrpo.official reads the problem as the second path part. A contest id is digits and an index starts with a letter, so
the joined id is unique (checked).

Line ends: open-r1 stores every official test with CRLF (885 of 885 problems; no example, no AOJ test). Codeforces ran
C programs on Windows, whose text-mode stdin turns CRLF into LF, so the accepted programs never saw the CR; on macOS a
program reading with getchar would. Tests are stored with LF (measured Nov 22), and a CR left after that stops the run."""
import hashlib
import itertools

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

log = structlog.get_logger("ggrpo")


def main() -> None:
    options = CliApp.run(Settings).codeforces
    if abs(sum(options.split.values()) - 1) > 1e-9:
        raise ValueError(f"codeforces.split must sum to 1: {options.split}")
    example = pl.element().struct
    problems = (
        pl.scan_parquet(options.problems.download() / "data")
        .filter(
            pl.col("input_mode") == "stdio",
            pl.col("generated_checker").is_null(),
            pl.col("interaction_format").is_null(),
            pl.col("official_tests_complete"),
            pl.col("official_tests").list.eval(example.field("output")).list.n_unique() >= 2,
            pl.col("examples").list.len() >= 1,
        )
        .select(
            problem_id=pl.col("id").str.replace("/", "", literal=True),
            source_id="id",
            aliases=pl.col("aliases").list.eval(pl.element().str.replace("/", "", literal=True)),
            rating="rating",
            year="contest_start_year",
            time_limit="time_limit",
            examples=pl.col("examples").list.eval(pl.struct(input=example.field("input").str.strip_chars(),
                                                            output=example.field("output").str.strip_chars())),
            tests=pl.col("official_tests").list.eval(pl.struct(
                input=example.field("input").str.replace_all("\r\n", "\n", literal=True),
                output=example.field("output").str.replace_all("\r\n", "\n", literal=True))),
        )
        .collect()
    )
    stray = problems.filter(pl.col("tests").list.eval(example.field("input").str.contains("\r", literal=True)).list.any()
                            | pl.col("examples").list.eval(example.field("input").str.contains("\r", literal=True)).list.any())
    if stray.height:
        raise ValueError(f"inputs with a CR outside a CRLF: {stray['problem_id'].to_list()[:10]}")
    if problems["problem_id"].is_duplicated().any():
        raise ValueError(f"problem ids collide without the slash: {problems.filter(pl.col('problem_id').is_duplicated())}")
    aliased = problems.select(pl.col("aliases").explode()).drop_nulls()["aliases"]
    if (shared := aliased.filter(aliased.is_in(problems["problem_id"].implode()))).len():
        raise ValueError(f"aliases of one another among the kept problems, group them first: {shared.to_list()}")

    solutions = (
        pl.scan_parquet(options.submissions.download() / "data")
        .filter(pl.col("programmingLanguage").is_in(options.languages), verdict="OK")
        .select(problem_id=pl.col("problem_id").str.replace("/", "", literal=True), solution_id="submission_id",
                c="source")
        .join(problems.lazy().select("problem_id"), on="problem_id", how="semi")
        .collect()
        .sort("problem_id", pl.col("solution_id").cast(pl.Int64))
        .with_columns(key=pl.col("c").str.replace_all(r"\s+", ""))
        .unique(["problem_id", "key"], keep="first", maintain_order=True)
        .drop("key")
    )
    solutions = (
        solutions.with_columns(rank=pl.Series([hashlib.sha256(f"{options.seed}:{solution_id}".encode()).hexdigest()
                                               for solution_id in solutions["solution_id"]], dtype=pl.String))
        .sort("problem_id", "rank")
        .group_by("problem_id", maintain_order=True)
        .agg(distinct=pl.len(), solutions=pl.struct("solution_id", "c").head(options.solutions))
    )

    # identical examples make one group, named by its smallest problem id
    problems = problems.join(solutions, on="problem_id", how="inner", validate="1:1").with_columns(
        group=pl.col("problem_id").min().over(
            pl.col("examples").list.eval(pl.concat_str(example.field("input"), pl.lit("\x1f"), example.field("output")))
            .list.sort().list.join("\x1e"))
    )
    quantile = pl.Series([int(hashlib.sha256(f"{options.seed}:{group}".encode()).hexdigest()[:15], 16) / 16**15
                          for group in problems["group"]], dtype=pl.Float64)
    problems = problems.with_columns(split=pl.coalesce(   # the first split whose cumulative fraction exceeds the quantile
        pl.when(quantile < bound).then(pl.lit(name))
        for name, bound in zip(options.split, itertools.accumulate(options.split.values()))
    )).sort("problem_id")
    if problems["split"].is_null().any():
        raise ValueError(f"groups past the last cumulative fraction: {problems.filter(pl.col('split').is_null())}")

    options.out.parent.mkdir(parents=True, exist_ok=True)
    problems.write_parquet(options.out)
    log.info("wrote", path=str(options.out), problems=problems.height,
             grouped=problems.filter(pl.col("group").is_duplicated()).height,
             splits=problems.group_by("split").agg(problems=pl.len(), groups=pl.col("group").n_unique(),
                                                   solutions=pl.col("solutions").list.len().sum())
             .sort("split").to_dicts())


if __name__ == "__main__":
    main()
