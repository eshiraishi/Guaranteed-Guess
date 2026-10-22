"""Run the paper's eval/eval_huamaneval.py as released, after set_seed(paper.seed), in the model-card environment
(uv sync --extra paper). With paper.sample = false the only change is `do_sample=False` next to `num_beams=8`.
Writes the script's eval_armv8_O2.json and predictions.jsonl (task_id, pred, ed) to paper.out."""
import json
import os
from pathlib import Path

import polars as pl
import structlog
from pydantic_settings import CliApp
from transformers import set_seed

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
source = settings.paper.script.read_text()
if not settings.paper.sample:
    if source.count("num_beams=8,") != 1:
        raise ValueError(f"expected one `num_beams=8,` in {settings.paper.script}")
    source = source.replace("num_beams=8,", "num_beams=8,\n        do_sample=False,")

log.info("paper.start", script=str(settings.paper.script), seed=settings.paper.seed, sample=settings.paper.sample,
         out=str(settings.paper.out))
settings.paper.out.mkdir(parents=True, exist_ok=True)
os.chdir(settings.paper.out)   # the script writes eval_armv8_O2.json to the working directory
set_seed(settings.paper.seed)
exec(compile(source, str(settings.paper.script.resolve()), "exec"), {"__name__": "__main__"})

# gg-bench-armv8-O2 file eval/problem{k+1}/code.c is humaneval/k (x86 and arm verified identical for all 164)
predictions = pl.DataFrame(json.loads(Path("eval_armv8_O2.json").read_text())).select(
    task_id=pl.format("humaneval/{}", pl.col("files").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1),
    pred="pred",
    ed="ed",
)
predictions.write_ndjson("predictions.jsonl")
log.info("paper.done", rows=predictions.height, out=str(settings.paper.out))
