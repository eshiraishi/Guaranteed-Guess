"""CodeNet RL task files (train, val, test) in the MBPP-C task format (task_id, arm_ref, prompt), with the problem's
samples as stdio tests instead of a C driver. A task is one solution whose reference arm passed the samples in the VM
build. Each split keeps at most codenet.cap[split] tasks per problem, chosen by sha256(seed:solution_id). An x86
prompt shared by two problems is dropped everywhere, so no prompt is in two splits."""
import hashlib
from collections import Counter

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
if set(settings.codenet.cap) != set(settings.codenet.tasks) or set(settings.codenet.tasks) != set(
    settings.codenet.split
):
    raise ValueError("codenet.cap, codenet.tasks and codenet.split must name the same splits")

solutions = pl.read_parquet(settings.codenet.solutions).filter(pl.col("split") != "sft_only")
built = pl.read_parquet(settings.codenet.built / "part-*.parquet")
if missing := solutions.join(built, on="solution_id", how="anti").height:
    raise ValueError(f"{missing} RL solutions are not in {settings.codenet.built} yet")

rank = pl.col("solution_id").map_elements(
    lambda solution_id: hashlib.sha256(f"{settings.codenet.seed}:{solution_id}".encode()).hexdigest(),
    return_dtype=pl.String,
)
tasks = (
    solutions.join(built.filter(ref_verdict="pass"), on="solution_id", validate="1:1")   # passing implies both built
    .join(pl.read_parquet(settings.codenet.problems).select("problem_id", "samples"), on="problem_id", validate="m:1")
    .sort("problem_id", rank)
    .unique(["problem_id", "x86"], keep="first", maintain_order=True)
    .filter(~pl.col("x86").is_duplicated())   # left: the same x86 under two problems
    .filter(pl.int_range(pl.len()).over("problem_id") < pl.col("split").replace_strict(settings.codenet.cap))
    .select(
        task_id=pl.format("codenet/{}/{}", "problem_id", "solution_id"),
        split="split",
        arm_ref="arm",
        tests="samples",
        prompt=pl.concat_list(pl.struct(role=pl.lit("user"), content=pl.format(settings.prompt.template, "x86"))),
    )
)
for split, path in settings.codenet.tasks.items():
    frame = tasks.filter(split=split).drop("split")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_ndjson(path)
    log.info("wrote", path=str(path), tasks=frame.height,
             problems=frame["task_id"].str.extract(r"codenet/([^/]+)/").n_unique())
log.info("references", verdicts=Counter(solutions.join(built, on="solution_id")["ref_verdict"].to_list()))
