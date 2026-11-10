"""The GRPO train file (Mac, --extra data, after ggrpo.decontam): the tasks of every file in rl.sources (MBPP-C train,
CodeNet train), except those with a drop row for an eval item in ggrpo.decontam's output, and those whose prompt or
reference answer is longer than rl.max_prompt_tokens or rl.max_answer_tokens under the base model's tokenizer (so
every kept task fits GRPO's own limits). Writes task_id and prompt, the columns trl reads."""
import polars as pl
import structlog
from huggingface_hub import hf_hub_download
from pydantic_settings import CliApp
from tokenizers import Tokenizer

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
tokenizer = Tokenizer.from_file(hf_hub_download(settings.rl.tokenizer, "tokenizer.json",
                                                revision=settings.rl.tokenizer_revision))


def count_tokens(texts: pl.Series) -> pl.Series:
    return pl.Series([len(encoding.ids) for encoding in tokenizer.encode_batch(texts.to_list())], dtype=pl.Int64)


tasks = pl.concat([pl.read_ndjson(path).select("task_id", "prompt", "arm_ref") for path in settings.rl.sources])
if tasks["task_id"].is_duplicated().any():
    raise ValueError("task ids must be unique across rl.sources")
contaminated = pl.read_parquet(settings.decontam.out).filter(   # eval items only: RL items exist to clean SFT (D5)
    "drop", ~pl.col("item_id").is_in(tasks["task_id"].implode()))
tasks = tasks.with_columns(
    source=pl.col("task_id").str.extract(r"^([^/]+)/"),
    contaminated=pl.col("task_id").is_in(contaminated["document_id"].implode()),
    prompt_tokens=count_tokens(tasks["prompt"].list.first().struct.field("content")),
    answer_tokens=count_tokens(pl.Series([settings.prompt.answer.format(arm) for arm in tasks["arm_ref"]])),
).with_columns(
    too_long=(pl.col("prompt_tokens") > settings.rl.max_prompt_tokens)
    | (pl.col("answer_tokens") > settings.rl.max_answer_tokens),
)
kept = tasks.filter(~pl.col("contaminated") & ~pl.col("too_long"))
settings.rl.out.parent.mkdir(parents=True, exist_ok=True)
kept.select("task_id", "prompt").write_ndjson(settings.rl.out)
counts = tasks.group_by("source").agg(
    tasks=pl.len(), contaminated=pl.col("contaminated").sum(), too_long=pl.col("too_long").sum(),
    kept=(~pl.col("contaminated") & ~pl.col("too_long")).sum(),
)
log.info("wrote", path=str(settings.rl.out), rows=kept.height, counts=counts.sort("source").to_dicts())
