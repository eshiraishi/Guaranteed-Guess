"""Score a file of ggrpo.sample rows written with sample.score = false, through the reward server, into the same row
format ggrpo.sample writes when it scores while sampling. Runs on the Mac, next to the VM, so the GPU hosts never
wait on the reward server (Nov 23)."""
import json

import httpx
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

log = structlog.get_logger("ggrpo")


def main() -> None:
    options = CliApp.run(Settings).score_samples
    rows = pl.read_ndjson(options.samples, infer_schema_length=None).sort("rank", "sample").to_dicts()
    if any("passed" in row for row in rows):
        raise ValueError(f"{options.samples} is already scored")
    results = [result for start in range(0, len(rows), options.batch)
               # the VM is next to the Mac, so a failure here is a real error: no retry
               for result in httpx.post(str(options.url), timeout=None,
                                        json=[{"task_id": row["task_id"], "completion": row["raw"]}
                                              for row in rows[start:start + options.batch]]).raise_for_status().json()]
    options.out.parent.mkdir(parents=True, exist_ok=True)
    partial = options.out.with_name(options.out.name + ".partial")
    partial.write_text("".join(json.dumps(row | {key: result[key] for key in ("verdict", "tests", "passed")}) + "\n"
                               for row, result in zip(rows, results, strict=True)))
    partial.rename(options.out)
    log.info("scored", samples=str(options.samples), rows=len(rows),
             pass_rate=round(sum(result["passed"] for result in results) / len(results), 4))


if __name__ == "__main__":
    main()
