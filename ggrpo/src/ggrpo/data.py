"""Build the train and dev (MBPP-C) and eval (HumanEval-C, the authors' data and tests) task files read by trl, the
generator and the reward server. Every task is task_id, x86, arm_ref, prompt and one C test driver."""
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
# Some tasks share the same x86 with different drivers: sample distinct programs so a prompt never lands on both sides
dev_x86 = (
    mbpp["x86"].unique(maintain_order=True).sample(fraction=settings.data.dev_fraction, seed=settings.data.split_seed)
).implode()
train, dev = mbpp.filter(~pl.col("x86").is_in(dev_x86)), mbpp.filter(pl.col("x86").is_in(dev_x86))

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

prompt = pl.concat_list(pl.struct(role=pl.lit("user"), content=pl.format(settings.prompt.template, "x86")))
for frame, path in [(train, settings.data.train), (dev, settings.data.dev), (humaneval, settings.data.eval)]:
    problems = frame.select(
        missing_driver=pl.col("driver").is_null().any(),
        duplicate_ids=pl.col("task_id").is_duplicated().any(),
    ).row(0, named=True)
    if any(problems.values()):
        raise ValueError(f"{path}: {problems}")
    frame.select("task_id", "arm_ref", "driver", prompt=prompt).write_ndjson(path)
    log.info("wrote", path=str(path), rows=frame.height)

# SFT baseline: the same train prompts, labelled with the reference ARM in GG's answer format (TRL prompt-completion)
sft = train.filter(~pl.col("task_id").is_in(settings.data.sft_exclude)).select(
    prompt=prompt,
    completion=pl.concat_list(pl.struct(role=pl.lit("assistant"), content=pl.format(settings.prompt.answer, "arm_ref"))),
)
sft.write_ndjson(settings.data.sft)
log.info("wrote", path=str(settings.data.sft), rows=sft.height, excluded=settings.data.sft_exclude)
