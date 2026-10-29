"""Export every run's eval rows from W&B to one CSV (config.toml [track].export): name, seed, step, metric, value.

Reads run histories, not W&B summaries: a summary is the last value logged, not the value at the last step.
"""
import polars as pl
import wandb

from ggrpo.settings import Settings

settings = Settings()

rows = [
    {"name": run.name, "seed": run.config.get("seed"), "step": row["train/global_step"],
     "metric": key.removeprefix("eval/"), "value": value}
    for run in wandb.Api(api_key=settings.wandb_api_key.get_secret_value()).runs(f"{settings.track.entity}/{settings.track.project}")
    for row in run.scan_history()
    for key, value in row.items() if key.startswith("eval/") and value is not None
]
settings.track.export.parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).sort("name", "seed", "metric", "step").write_csv(settings.track.export)
