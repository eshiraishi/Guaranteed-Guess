"""The RL pool and the SFT-scale arms' data, from the census (prereg.md §2; Mac or training host, after ggrpo.sample's
census and once C-0's step time fixes pool.cap):
- pool.out (task_id, prompt), GRPO's train file: the first pool.cap tasks in census order whose samples disagree
  (0 < passes < samples);
- pool.references, SFT on pool references' dataset: those tasks' compiler references;
- pool.offline_rft, offline RFT's dataset: those tasks' passing census samples.
Both SFT folders hold a train.parquet with TRL's prompt and completion columns, as data/sft/redo_200k. Logs the online
step count (pool.passes passes of pool.prompts_per_step prompts) and the longest passing census completion, for the
completion cap rule (8,192 tokens if none above 8k passes, else 16,384)."""
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.sample import keep_prefix
from ggrpo.settings import Settings

log = structlog.get_logger("ggrpo")


def select_pool(census: pl.DataFrame, cap: int) -> pl.DataFrame:
    """task_id of the first cap tasks in census order with 0 < passes < their sample count, from its scored prefix."""
    per_task = keep_prefix(census).group_by("task_id").agg(rank=pl.col("rank").first(), n=pl.len(), passes=pl.col("passed").sum())
    if per_task["n"].n_unique() != 1:
        raise ValueError(f"tasks have different sample counts: {sorted(per_task['n'].unique().to_list())}")
    mixed = per_task.filter((pl.col("passes") > 0) & (pl.col("passes") < pl.col("n"))).sort("rank")
    if mixed.height < cap:
        raise ValueError(f"the census found {mixed.height} mixed tasks, fewer than the cap {cap}: census more tasks")
    return mixed.head(cap).select("task_id")


def to_sft(frame: pl.DataFrame, answer: pl.Expr) -> pl.DataFrame:
    """(prompt, answer text) -> TRL prompt-completion rows, the assistant turn as ggrpo.sft writes it."""
    return frame.select("prompt", completion=pl.concat_list(pl.struct(role=pl.lit("assistant"), content=answer)))


def main() -> None:
    settings = CliApp.run(Settings)
    config = settings.pool
    if config.cap <= 0 or config.cap % config.prompts_per_step:
        raise SystemExit(f"pool.cap must be a positive multiple of {config.prompts_per_step}, got {config.cap}")
    census = pl.read_ndjson(config.census)
    pool = select_pool(census, config.cap)
    tasks = pl.read_ndjson(settings.rl.out).select("task_id", "prompt").join(pool, on="task_id", validate="1:1")
    if tasks.height != config.cap:
        raise ValueError(f"{config.cap - tasks.height} pool tasks are not in {settings.rl.out}")
    config.out.parent.mkdir(parents=True, exist_ok=True)
    pool.join(tasks, on="task_id", maintain_order="left").write_ndjson(config.out)   # census order

    references = pl.concat([pl.read_ndjson(path).select("task_id", "arm_ref") for path in settings.rl.sources])
    passing = census.filter("passed").select("task_id", "raw")
    for folder, rows, answer in [
        (config.references, tasks.join(references, on="task_id", validate="1:1"),
         pl.format(settings.prompt.answer, "arm_ref")),
        (config.offline_rft, tasks.join(passing, on="task_id"), pl.col("raw")),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        to_sft(rows, answer).write_parquet(folder / "train.parquet")
        log.info("wrote", path=str(folder / "train.parquet"), rows=rows.height)
    longest = census.filter("passed")["new_tokens"].max()
    log.info("pool", path=str(config.out), tasks=config.cap,
             steps=config.passes * (config.cap // config.prompts_per_step), longest_passing_completion=longest,
             completion_cap=16384 if longest > 8192 else 8192)


if __name__ == "__main__":
    main()
