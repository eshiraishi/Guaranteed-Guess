"""Re-score the completions that passed their sample tests against the differential suite (prereg.md §4), to measure
how often the sample tests pass a wrong program. A completion that already failed the sample tests cannot pass a
superset of them, so only the passes are re-run, and each stops at its first failing test."""
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import structlog
from pydantic import BaseModel
from pydantic_settings import CliApp

from ggrpo.harness import execute, link, normalize
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")


class Recheck(BaseModel):
    task_id: str
    sample: int | None         # the sample index in a multi-sample file (E2), None in a one-per-task file (beam)
    tests: int                 # tests in the suite this completion was run against
    passed: bool
    failed_at: int | None      # index of the first failing test, None when every test passed
    verdict: str


def check_suite(task_id: str, sample: int | None, asm: str, tests: list[dict[str, str]]) -> Recheck:
    """Run the suite in order and stop at the first failure."""
    with tempfile.TemporaryDirectory(prefix="gg-recheck-") as tmp:
        work = Path(tmp).resolve()
        if failure := link(work, asm, [], 0.0):
            return Recheck(task_id=task_id, sample=sample, tests=len(tests), passed=False, failed_at=0, verdict=failure.verdict)
        for position, test in enumerate(tests):
            finished = execute(work, test["input"] + "\n")
            if finished is None:
                return Recheck(task_id=task_id, sample=sample, tests=len(tests), passed=False, failed_at=position, verdict="timeout")
            if finished.returncode < 0:
                return Recheck(task_id=task_id, sample=sample, tests=len(tests), passed=False, failed_at=position, verdict="crash")
            if normalize(finished.stdout) != normalize(test["output"]):
                return Recheck(task_id=task_id, sample=sample, tests=len(tests), passed=False, failed_at=position,
                               verdict="wrong_output")
    return Recheck(task_id=task_id, sample=sample, tests=len(tests), passed=True, failed_at=None, verdict="pass")


def check_payload(payload: tuple[str, int | None, str, list[dict[str, str]]]) -> Recheck:
    return check_suite(*payload)


def load_suites(path: Path, min_kept: int) -> dict[str, list[dict[str, str]]]:
    """Each usable problem's suite, keyed by task id; unusable problems are left out and keep their sample tests."""
    suites = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["kept"] >= min_kept and row["distinct_outputs"] >= 2:
            suites[row["task_id"]] = row["tests"]
    return suites


def main() -> None:
    options = CliApp.run(Settings).recheck
    # The scored files store the raw generation under "completion"; the reward server extracts the assembly itself.
    scored = pl.read_ndjson(options.scored)
    if "sample" not in scored.columns:
        scored = scored.with_columns(sample=pl.lit(None, dtype=pl.Int64))
    scored = scored.select(
        "task_id", "sample", "passed", completion=pl.col(options.column).str.extract(settings.prompt.extract, 1))
    suites = load_suites(options.suites, settings.differential.min_kept)
    passes = scored.filter("passed")
    payloads = [(row["task_id"], row["sample"], row["completion"], suites[row["task_id"]])
                for row in passes.iter_rows(named=True) if row["task_id"] in suites]
    covered = {payload[0] for payload in payloads}
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        results = list(pool.map(check_payload, payloads, chunksize=1))
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text("".join(result.model_dump_json() + "\n" for result in results))
    survived = sum(result.passed for result in results)
    log.info(
        "recheck",
        scored=str(options.scored),
        sample_passes=passes.height,
        with_usable_suite=len(covered),
        rechecked=len(payloads),
        no_usable_suite=passes.height - len(payloads),
        survived_differential=survived,
        false_passes=len(results) - survived,
        verdicts=Counter(result.verdict for result in results),
    )


if __name__ == "__main__":
    main()
