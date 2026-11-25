"""Export every run's eval rows from W&B to one CSV (config.toml [track].export): name, seed, step, metric, value.

Reads run histories, not W&B summaries: a summary is the last value logged, not the value at the last step.

Only the pass@1 metrics registered in [track] are exported, at the steps registered for the run that holds them. The
trainer logs its own eval/ keys (loss, entropy, runtime, num_tokens), which are not pass rates, and an eval logged
into a run while it was still training can carry the trainer's step instead of its own (Nov 16, sft-full-seed0).
"""
import polars as pl
import wandb

from ggrpo.settings import Settings

settings = Settings()
metrics = {metric for run in settings.track.runs.values() for files in run.evals.values() for metric in files}
steps = {run_id: set(run.evals) for run_id, run in settings.track.runs.items()}


def keeps(run_id: str, step: int | None, metric: str) -> bool:
    """A registered pass@1 metric, at a step registered for this run if the run is one we track."""
    return metric in metrics and (run_id not in steps or step in steps[run_id])


rows = [
    {"name": run.name, "seed": run.config.get("seed"), "step": row["train/global_step"],
     "metric": key.removeprefix("eval/"), "value": value}
    for run in wandb.Api(api_key=settings.wandb_api_key.get_secret_value()).runs(f"{settings.track.entity}/{settings.track.project}")
    for row in run.scan_history()
    for key, value in row.items()
    if key.startswith("eval/") and value is not None
    and keeps(run.id, row["train/global_step"], key.removeprefix("eval/"))
]
settings.track.export.parent.mkdir(parents=True, exist_ok=True)
# a rerun of ggrpo.track can log a value twice when W&B has not yet served the row it would have skipped
pl.DataFrame(rows).unique().sort("name", "seed", "metric", "step").write_csv(settings.track.export)
