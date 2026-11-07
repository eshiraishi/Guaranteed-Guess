"""Build the CodeNet pool in the macOS 13 VM (the GG toolchain). Each distinct solution is compiled to x86 and arm
with GG's commands (build.x86, build.arm at build.opt). For RL problems the reference arm must also pass the
statement's samples in stdio mode, which catches toolchain differences, special judges and wrong "accepted" code.
RL problems go first. Writes codenet.built as parquet parts of codenet.chunk solutions and skips the parts that exist,
so a stopped build resumes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.build import Task
from ggrpo.harness import check_stdio
from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
schema = {"solution_id": pl.String, "x86": pl.String, "arm": pl.String, "ref_verdict": pl.String,
          "ref_stderr": pl.String}


def build_solution(solution_id: str, c: str, samples: list[dict[str, str]] | None) -> dict[str, str | None]:
    """x86 and arm of one solution (None where clang fails); the reference verdict only when samples are given."""
    x86, arm = [Task.compile(c, cmd) for cmd in (settings.build.x86, settings.build.arm)]
    ref = check_stdio(arm, samples) if samples is not None and arm is not None else None
    return {"solution_id": solution_id, "x86": x86, "arm": arm, "ref_verdict": ref and ref.verdict,
            "ref_stderr": ref and ref.stderr}


if __name__ == "__main__":   # pool workers import this module
    problems = pl.read_parquet(settings.codenet.problems).select("problem_id", "samples")
    solutions = (
        pl.read_parquet(settings.codenet.solutions)
        .join(problems, on="problem_id", how="left", validate="m:1")
        .with_columns(samples=pl.when(pl.col("split") != "sft_only").then("samples"))   # references checked on RL only
        .sort(pl.col("split") == "sft_only", "problem_id", "solution_id")
    )
    settings.codenet.built.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(settings.build.workers) as pool:
        for start in range(0, solutions.height, settings.codenet.chunk):
            part = settings.codenet.built / f"part-{start // settings.codenet.chunk:05d}.parquet"
            if part.exists():
                continue
            chunk = solutions.slice(start, settings.codenet.chunk)
            rows = list(pool.map(build_solution, chunk["solution_id"], chunk["c"], chunk["samples"].to_list(),
                                 chunksize=8))
            built = pl.DataFrame(rows, schema=schema)
            built.write_parquet(part.with_suffix(".tmp"))
            part.with_suffix(".tmp").replace(part)
            log.info("part", path=str(part), rows=built.height, x86=built["x86"].is_not_null().sum(),
                     arm=built["arm"].is_not_null().sum(), ref=Counter(built["ref_verdict"].drop_nulls().to_list()))
    built = pl.read_parquet(settings.codenet.built / "part-*.parquet")
    if built.height != solutions.height or set(built["solution_id"]) != set(solutions["solution_id"]):
        raise ValueError(f"{settings.codenet.built} holds {built.height} rows for {solutions.height} solutions")
    log.info("built", solutions=built.height, x86=built["x86"].is_not_null().sum(),
             arm=built["arm"].is_not_null().sum(), ref=Counter(built["ref_verdict"].drop_nulls().to_list()))
