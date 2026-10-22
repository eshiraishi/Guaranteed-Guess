"""Score completions through the reward server; log pass@1 on all tasks and on reference-valid tasks."""
from collections import Counter

import httpx
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
completions = pl.read_ndjson(settings.score.predictions).select("task_id", completion=settings.score.column)
scored = (
    pl.read_ndjson(settings.score.tasks)
    .select("task_id")
    .join(
        completions.hstack(pl.DataFrame(
            httpx.post(str(settings.server.url), json=completions.to_dicts(), timeout=None).raise_for_status().json()
        )),
        on="task_id",
        how="left",
        validate="1:1",
    )
)
if missing := scored.filter(pl.col("verdict").is_null())["task_id"].to_list():
    raise SystemExit(f"no prediction for {missing}")
settings.score.out.parent.mkdir(parents=True, exist_ok=True)
scored.write_ndjson(settings.score.out)

valid = (
    pl.read_ndjson(settings.score.reference).filter("passed")["task_id"]
    if settings.score.reference.exists()
    else scored["task_id"]
)
log.info(
    "pass@1",
    predictions=str(settings.score.predictions),
    verdicts=Counter(scored["verdict"].to_list()),
    summary=scored.select(
        passed=pl.col("passed").sum(),
        total=pl.len(),
        valid_passed=pl.col("passed").filter(pl.col("task_id").is_in(valid.implode())).sum(),
        valid_total=pl.col("task_id").is_in(valid.implode()).sum(),
    ).row(0, named=True),
)
