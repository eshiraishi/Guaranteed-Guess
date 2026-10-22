"""Build the train (MBPP-C) and eval (HumanEval-C, the authors' data and tests) task files read by trl, the generator
and the reward server. Every task is task_id, x86, arm_ref, prompt and one C test driver."""
import polars as pl
import structlog

from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")

mbpp = (
    pl.read_parquet(settings.data.mbpp)
    .filter(ref_verdict="pass", neg_verdict="link_err")
    .select("task_id", "x86", "arm_ref", "driver")
)
if settings.data.train_ids:
    mbpp = mbpp.filter(pl.col("task_id").is_in(settings.data.train_ids.read_text().split()))

# gg-bench file eval/problem{n}/code.c is humaneval/{n-1}; its tests come from the authors' asmbench_source, by file
humaneval = (
    pl.read_parquet(settings.data.humaneval)
    .join(pl.read_parquet(settings.data.humaneval_tests), on="file", how="left", validate="1:1")
    .select(
        task_id=pl.format("humaneval/{}", pl.col("file").str.extract(r"problem(\d+)/").cast(pl.Int64) - 1),
        x86="x86",
        arm_ref="arm",
        driver="test",
    )
)

for frame, path in [(mbpp, settings.data.train), (humaneval, settings.data.eval)]:
    problems = frame.select(
        missing_driver=pl.col("driver").is_null().any(),
        duplicate_ids=pl.col("task_id").is_duplicated().any(),
    ).row(0, named=True)
    if any(problems.values()):
        raise ValueError(f"{path}: {problems}")
    frame.select(
        "task_id",
        "arm_ref",
        "driver",
        prompt=pl.concat_list(pl.struct(role=pl.lit("user"), content=pl.format(settings.prompt.template, "x86"))),
    ).write_ndjson(path)
    log.info("wrote", path=str(path), rows=frame.height)
