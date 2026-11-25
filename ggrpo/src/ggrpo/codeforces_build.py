"""Build the Codeforces eval pool in the macOS 13 VM (the GG toolchain), like ggrpo.codenet_build. Each solution that
ggrpo.codeforces kept is compiled to x86 and arm with GG's commands (build.x86, build.arm at build.opt), and each built
arm runs the statement's examples and every official test, sandboxed under the harness limits. Per test it records P
(pass: equal up to trailing whitespace, as the scorers compare), S (the same whitespace-separated tokens: Codeforces'
default checker compares tokens), K (the same tokens up to letter case, which most Codeforces checkers accept), W
(wrong output), C (crash) or T (timeout, which ends the run, as in harness.check_stdio), and the CPU seconds of the
slowest test; ggrpo.codeforces_tasks picks the reference, applies the timing margin and checks agreement from
them. Runs only accepted human solutions, never a model's output.

Writes codeforces.built as parquet parts of codeforces.chunk problems and skips the parts that exist, so a stopped build
resumes; then codeforces.candidates: every solution that built and passed every test, in task format with its C, the
items of ggrpo.decontam's Codeforces run."""
import resource
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.build import Task
from ggrpo.harness import execute, link, normalize
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")
schema = {"problem_id": pl.String, "solution_id": pl.String, "rank": pl.Int64, "x86": pl.String, "arm": pl.String,
          "on_examples": pl.String, "on_tests": pl.String, "cpu": pl.Float64}


def run_tests(asm: str, tests: list[dict[str, str]]) -> tuple[str, float]:
    """Outcome per test (P, S, K, W, C, T; a timeout ends the run) and the CPU seconds of the slowest test; ("L", 0.0) when
    the asm does not assemble and link. Inputs get their final newline back, as in harness.check_stdio and
    recheck.check_suite. CPU time is the child's, read around each run, so a caller runs one program at a time."""
    with tempfile.TemporaryDirectory(prefix="gg-cf-") as tmp:
        work = Path(tmp).resolve()
        if link(work, asm, [], 0.0) is not None:
            return "L", 0.0
        outcomes, slowest = "", 0.0
        for test in tests:
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            finished = execute(work, test["input"] + "\n")
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            slowest = max(slowest, after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime)
            if finished is None or -finished.returncode == signal.SIGXCPU:
                return outcomes + "T", slowest
            printed, expected = normalize(finished.stdout), normalize(test["output"])
            outcomes += ("C" if finished.returncode < 0 else "P" if printed == expected
                         else "S" if printed.split() == expected.split()
                         else "K" if printed.casefold().split() == expected.casefold().split() else "W")
    return outcomes, slowest


def build_problem(problem_id: str, examples: list[dict[str, str]], tests: list[dict[str, str]],
                  solutions: list[dict[str, str]]) -> list[dict]:
    """One row per solution, in pick order: its x86 and arm (None where clang fails) and, for a built arm, the
    outcomes on the examples and on the official tests and the slowest test's CPU seconds."""
    rows = []
    for rank, solution in enumerate(solutions):
        x86, arm = [Task.compile(solution["c"], cmd) for cmd in (settings.build.x86, settings.build.arm)]
        row = {"problem_id": problem_id, "solution_id": solution["solution_id"], "rank": rank, "x86": x86, "arm": arm,
               "on_examples": None, "on_tests": None, "cpu": None}
        if arm is not None:
            (row["on_examples"], examples_cpu), (row["on_tests"], tests_cpu) = run_tests(arm, examples), run_tests(arm, tests)
            row["cpu"] = max(examples_cpu, tests_cpu)
        rows.append(row)
    return rows


def main() -> None:
    options = CliApp.run(Settings).codeforces
    problems = pl.read_parquet(options.out)
    options.built.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(settings.build.workers) as pool:
        for start in range(0, problems.height, options.chunk):
            part = options.built / f"part-{start // options.chunk:05d}.parquet"
            if part.exists():
                continue
            chunk = problems.slice(start, options.chunk)
            rows = [row for problem in pool.map(build_problem, chunk["problem_id"], chunk["examples"].to_list(),
                                                chunk["tests"].to_list(), chunk["solutions"].to_list())
                    for row in problem]
            built = pl.DataFrame(rows, schema=schema)
            built.write_parquet(part.with_suffix(".tmp"))
            part.with_suffix(".tmp").replace(part)
            log.info("part", path=str(part), problems=chunk.height, solutions=built.height,
                     x86=built["x86"].is_not_null().sum(), arm=built["arm"].is_not_null().sum(),
                     pass_examples=built["on_examples"].str.contains(r"^P+$").sum(),
                     pass_tests=built["on_tests"].str.contains(r"^P+$").sum())

    built = pl.read_parquet(options.built / "part-*.parquet")
    expected = dict(zip(problems["problem_id"], problems["solutions"].list.len()))
    if dict(built.group_by("problem_id").agg(pl.len()).iter_rows()) != expected:
        raise ValueError(f"{options.built} does not hold every kept solution of {options.out}: delete it and rerun")
    solutions = problems.select("problem_id", "solutions").explode("solutions").select(
        "problem_id", solution_id=pl.col("solutions").struct.field("solution_id"), c=pl.col("solutions").struct.field("c")
    )
    every_test_passed = [   # a timeout ends a run early, so a full-length string of P is a pass
        pl.col(outcomes).str.contains(r"^P+$") & (pl.col(outcomes).str.len_chars() == pl.col(tests).list.len())
        for outcomes, tests in [("on_examples", "examples"), ("on_tests", "tests")]
    ]
    candidates = (
        built.join(solutions, on=["problem_id", "solution_id"], validate="1:1")
        .join(problems.select("problem_id", "split", "examples", "tests"), on="problem_id", validate="m:1")
        .filter(pl.col("x86").is_not_null(), *every_test_passed)
        .select(
            task_id=pl.format("codeforces/{}/{}", "problem_id", "solution_id"),
            split="split",
            arm_ref="arm",
            tests="examples",
            prompt=pl.concat_list(pl.struct(role=pl.lit("user"), content=pl.format(settings.prompt.template, "x86"))),
            c="c",
        )
    )
    options.candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_ndjson(options.candidates)
    log.info("built", problems=problems.height, solutions=built.height, x86=built["x86"].is_not_null().sum(),
             arm=built["arm"].is_not_null().sum(), candidates=candidates.height,
             problems_with_candidates=candidates["task_id"].str.extract(r"^codeforces/([^/]+)/", 1).n_unique(),
             path=str(options.candidates))


if __name__ == "__main__":   # pool workers import this module
    main()
