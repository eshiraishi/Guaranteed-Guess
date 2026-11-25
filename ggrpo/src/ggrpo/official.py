"""The official pass (prereg.md §4, Nov 22 amendments): a completion passes when it passes its sample tests and, on an
AOJ or Codeforces problem, that problem's official tests too. Reads a file already scored on sample tests (its
`passed` column), so the sample verdicts stay the registered ones, and runs the official tests only on the
sample-test passes of AOJ and Codeforces problems. Writes one row per completion of such a problem, so pass@1 and
pass@k on those subsets come straight from the output."""
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import structlog
from pydantic import BaseModel
from pydantic_settings import CliApp

from ggrpo.calibration import load_official
from ggrpo.recheck import check_suite
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")


class OfficialVerdict(BaseModel):
    task_id: str
    sample: int
    sample_passed: bool
    official_passed: bool
    failed_at: int | None      # index of the first failing official test; None when it passed or was not run
    verdict: str               # the official-test verdict, or "sample_fail" when the official tests were not run


def check_official(payload: tuple) -> OfficialVerdict:
    task_id, sample, sample_passed, asm, tests = payload
    if not sample_passed:
        return OfficialVerdict(task_id=task_id, sample=sample, sample_passed=False, official_passed=False,
                               failed_at=None, verdict="sample_fail")
    result = check_suite(task_id, sample, asm, tests)
    return OfficialVerdict(task_id=task_id, sample=sample, sample_passed=True, official_passed=result.passed,
                           failed_at=result.failed_at, verdict=result.verdict)


def main() -> None:
    options = CliApp.run(Settings).official
    official = {}
    for path in options.official:   # AOJ and Codeforces ids cannot collide (p01234 vs 755D); checked anyway
        tests = load_official(path)
        if shared := sorted(set(official) & set(tests)):
            raise ValueError(f"{path} repeats problems of an earlier official file: {shared[:10]}")
        official |= tests
    frame = pl.read_ndjson(options.scored, infer_schema_length=None)
    frame = frame.select(
        "task_id",
        "passed",
        # Extraction is idempotent on text that carries no armasm fence, so it runs whichever column holds the output.
        completion=pl.col(options.column).str.extract(settings.prompt.extract, 1),
        sample=pl.col("sample").cast(pl.Int64) if "sample" in frame.columns else pl.lit(0),
        problem=pl.col("task_id").str.split("/").list.get(1),
    ).filter(pl.col("problem").is_in(list(official)))
    if frame["passed"].null_count():
        raise ValueError(f"{options.scored}: rows without a sample-test verdict")
    payloads = [(row["task_id"], row["sample"], row["passed"], row["completion"], official[row["problem"]])
                for row in frame.iter_rows(named=True)]
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        results = list(pool.map(check_official, payloads, chunksize=1))
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text("".join(result.model_dump_json() + "\n" for result in results))
    log.info(
        "official",
        scored=str(options.scored),
        tasks=len({result.task_id for result in results}),
        completions=len(results),
        sample_passes=sum(result.sample_passed for result in results),
        official_passes=sum(result.official_passed for result in results),
    )


if __name__ == "__main__":
    main()
