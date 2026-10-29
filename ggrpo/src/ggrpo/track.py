"""Log scored evals into W&B runs at their training step (config.toml [track]).

`uv run python -m ggrpo.track`, with WANDB_API_KEY in .env. Every value is pass@1.
(step, metric) pairs a run already has are skipped, so rerun it after listing a new file.
"""
from pathlib import Path

import polars as pl
import wandb

from ggrpo.settings import Settings

settings = Settings()


def read_tasks(path: Path) -> pl.DataFrame:
    """One row per task: task_id and the fraction of its rows that pass (1 row per task for beam/greedy, 8 for sample8)."""
    return (
        pl.read_ndjson(path)
        .rename({"bench": "task_id"}, strict=False)   # BringUp names its tasks `bench`
        .filter(~pl.col("task_id").is_in(settings.track.exclude))
        .group_by("task_id")
        .agg(passed=pl.col("verdict").eq("pass").mean())
    )


def read_pass_rate(path: Path) -> float:
    """Mean over tasks of each task's passing fraction."""
    return read_tasks(path)["passed"].mean()


if __name__ == "__main__":
    api_key = settings.wandb_api_key.get_secret_value()
    for run_id, run in settings.track.runs.items():
        if not any(path.exists() for files in run.evals.values() for path in files.values()):
            continue   # nothing scored yet: don't open an empty run before its training is copied or logged
        with wandb.init(entity=settings.track.entity, project=settings.track.project, id=run_id, name=run.name,
                        resume="allow", settings=wandb.Settings(api_key=api_key)) as wb:
            wb.define_metric("eval/*", step_metric="train/global_step")   # TRL's x-axis, so evals line up with training
            logged = {(row.get("train/global_step"), key)
                      for row in wandb.Api(api_key=api_key).run(wb.path).scan_history() for key in row}
            for step, files in sorted(run.evals.items()):
                metrics = {f"eval/{metric}": read_pass_rate(path) for metric, path in files.items()
                           if path.exists() and (step, f"eval/{metric}") not in logged}   # registered before it lands
                if metrics:
                    wb.log({"train/global_step": step} | metrics)
