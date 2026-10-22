"""Reward server (macOS 13 VM): POST /score [{task_id, completion}] -> one harness Result per item.
404 for unknown task ids; 503 if a driver failed to build, so training stops instead of learning from it."""
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ggrpo.harness import Result, check
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")
tasks = pl.concat([pl.read_ndjson(settings.data.train), pl.read_ndjson(settings.data.eval)]).select("task_id", "driver")
pool = ProcessPoolExecutor(settings.server.workers)   # processes, not threads: preexec_fn is unsafe in threads
app = FastAPI()


class Item(BaseModel):
    task_id: str
    completion: str


@app.post("/score")
def score(items: list[Item]) -> list[Result]:
    start = time.monotonic()
    batch = (
        pl.DataFrame([item.model_dump() for item in items], schema={"task_id": pl.String, "completion": pl.String})
        .join(tasks, on="task_id", how="left", maintain_order="left")
        .with_columns(asm=pl.col("completion").str.extract(settings.prompt.extract, 1))
    )
    if unknown := batch.filter(pl.col("driver").is_null())["task_id"].to_list():
        raise HTTPException(404, f"unknown task ids: {unknown}")
    results = list(pool.map(check, batch["asm"], batch["driver"]))
    log.info("scored", items=len(items), seconds=round(time.monotonic() - start, 2),
             verdicts=Counter([result.verdict for result in results]))
    if infra := [result.stderr for result in results if result.verdict == "infra_err"]:
        raise HTTPException(503, infra)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, access_log=False)
