"""Copy a run trained on Trackio (before the W&B switch, Oct 23) to W&B: TRL config, training metrics, profiling timers,
completion tables and Trackio's GPU monitor.

The W&B name is config.toml [track.runs.RUN_ID].name. Reads a backup of runs/trackio/cs8001.db, never the live file.
Trackio's own step counts log calls, so every row gets the training step it belongs to: a profiling row and a GPU
sample the update in progress, a completion table its `step` column. A run resumed from a checkpoint repeats the
steps after it: the rows logged before the resume are dropped, as the trainer discarded them.
A run already in W&B with training rows (grpo-seed0, copied Oct 23 with those only) gets the rest appended; a run that
only has evals gets everything; a second copy of the same run is refused.
usage (from ggrpo/): uv run python ops/migrate/trackio_to_wandb.py DB TRACKIO_NAME RUN_ID
"""
import bisect
import json
import sqlite3
import sys

import wandb

from ggrpo.settings import Settings

settings = Settings()
db_path, trackio_name, run_id = sys.argv[1:]
name = settings.track.runs[run_id].name
api_key = settings.wandb_api_key.get_secret_value()
db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
(config,) = db.execute("select config from configs where run_name = ?", (trackio_name,)).fetchone()


def cell(value):
    return value if isinstance(value, int | float | str) else json.dumps(value)   # chat messages as JSON text


rows, pending, ends = [], [], []   # rows: (is_training_row, row); pending: profiling rows before their update's row
for timestamp, metrics in db.execute("select timestamp, metrics from metrics where run_name = ? order by id",
                                     (trackio_name,)):
    row = json.loads(metrics)
    if "completions" in row:
        table = row["completions"]["_value"]
        rows.append((False, {"train/global_step": table[0]["step"], "completions": wandb.Table(
            columns=list(table[0]), data=[[cell(v) for v in r.values()] for r in table])}))
    elif "train/global_step" in row:
        step = row["train/global_step"]
        if ends and step < ends[-1][1]:
            rows = [(t, r) for t, r in rows if r["train/global_step"] < step]
            ends = [(t, s) for t, s in ends if s < step]
        rows += [(False, p | {"train/global_step": step}) for p in pending] + [(True, row)]
        pending = []
        ends.append((timestamp, step))
    else:
        pending.append(row)
rows += [(False, p | {"train/global_step": ends[-1][1]}) for p in pending]   # after the last update
# a GPU sample belongs to the first update that ends after it (ISO timestamps in UTC sort as text)
gpu = [(False, json.loads(m) | {"train/global_step": ends[min(bisect.bisect_left(ends, (t,)), len(ends) - 1)][1]})
       for t, m in db.execute("select timestamp, metrics from system_metrics where run_name = ? order by id",
                              (trackio_name,))]

try:
    existing = wandb.Api(api_key=api_key).run(f"{settings.track.entity}/{settings.track.project}/{run_id}")
except wandb.errors.CommError:
    existing = None
if existing and existing.summary.get("migrate/extras"):
    sys.exit(f"{run_id} is already copied")
trained = existing and "train/loss" in existing.summary
with wandb.init(entity=settings.track.entity, project=settings.track.project, id=run_id, name=name,
                resume="must" if existing else "never",
                # run_name was the old label ("full", "baseline, seed=1", "SFT, seed=0"); the backup keeps it
                config={k: v for k, v in json.loads(config).items() if not k.startswith("_")} | {"run_name": name},
                settings=wandb.Settings(api_key=api_key)) as run:
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")   # as transformers' W&B callback does
    for training, row in sorted(rows + gpu, key=lambda r: r[1]["train/global_step"]):
        if not (training and trained):   # the Oct 23 copy has the training rows
            run.log({k: v for k, v in row.items() if isinstance(v, int | float | wandb.Table)})
    run.summary["migrate/extras"] = 1
