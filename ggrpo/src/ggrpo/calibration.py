"""Compare the differential suite with AOJ's official tests, on the validation problems that have both
(prereg.md §4, Differential tests). Among completions that pass their problem's sample tests, it reports the share
of AOJ's rejections the differential suite catches and the share of AOJ's acceptances it wrongly rejects: the
prereg's gate on using the suite for E1 and E2."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import structlog
from pydantic import BaseModel
from pydantic_settings import CliApp

from ggrpo.recheck import check_suite
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")


class Comparison(BaseModel):
    task_id: str
    sample: int
    official_passed: bool
    differential_passed: bool
    official_tests: int
    differential_tests: int


def compare(payload: tuple) -> Comparison | None:
    """None when the completion fails its sample tests, which the prereg's gate excludes."""
    task_id, sample, asm, samples, official, differential = payload
    if not check_suite(task_id, None, asm, samples).passed:
        return None
    return Comparison(
        task_id=task_id,
        sample=sample,
        official_passed=check_suite(task_id, None, asm, official).passed,
        differential_passed=check_suite(task_id, None, asm, differential).passed,
        official_tests=len(official),
        differential_tests=len(differential),
    )


def load_suites(path, min_kept: int) -> dict[str, dict]:
    """Every usable problem's suite record, keyed by task id."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["task_id"]: row for row in rows if row["kept"] >= min_kept and row["distinct_outputs"] >= 2}


def load_official(path) -> dict[str, list[dict[str, str]]]:
    """Official tests keyed by the problem part of a task id: AOJ's file names it codenet_id (its problem_id is AOJ's
    own id), Codeforces' file (ggrpo.codeforces_tasks) problem_id."""
    frame = pl.read_parquet(path)
    key = "codenet_id" if "codenet_id" in frame.columns else "problem_id"
    return dict(zip(frame[key].to_list(), [[dict(test) for test in tests] for tests in frame["tests"].to_list()]))


def main() -> None:
    options = CliApp.run(Settings).calibration
    frame = pl.read_ndjson(options.completions)
    # Extraction is idempotent on text that carries no armasm fence, so it runs whichever column holds the output.
    frame = frame.select(
        "task_id",
        completion=pl.col(options.column).str.extract(settings.prompt.extract, 1),
        sample=pl.col("sample") if "sample" in frame.columns else pl.lit(0),
    )
    suites = load_suites(options.suites, settings.differential.min_kept)
    official = load_official(options.official)
    payloads = []
    for row in frame.iter_rows(named=True):
        problem = row["task_id"].split("/")[1]
        suite = suites.get(row["task_id"])
        if suite is None or problem not in official:
            continue
        payloads.append((row["task_id"], row["sample"], row["completion"], suite["tests"][:suite["samples"]],
                         official[problem], suite["tests"][suite["samples"]:]))
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        results = [result for result in pool.map(compare, payloads, chunksize=1) if result is not None]
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text("".join(result.model_dump_json() + "\n" for result in results))

    rejected = [result for result in results if not result.official_passed]
    accepted = [result for result in results if result.official_passed]
    caught = sum(1 for result in rejected if not result.differential_passed)
    wrongly = sum(1 for result in accepted if not result.differential_passed)
    log.info(
        "calibration",
        completions=str(options.completions),
        eligible=len(payloads),
        sample_test_passes=len(results),
        problems=len({result.task_id for result in results}),
        official_rejections=len(rejected),
        caught_by_differential=caught,
        sensitivity=round(caught / len(rejected), 4) if rejected else None,
        official_acceptances=len(accepted),
        wrongly_rejected=wrongly,
        false_rejection_rate=round(wrongly / len(accepted), 4) if accepted else None,
        tests=Counter(result.differential_tests for result in results).most_common(3),
    )


if __name__ == "__main__":
    main()
