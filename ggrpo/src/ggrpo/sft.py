"""SFT train and val files (Mac, --extra data, after ggrpo.decontam) in the SFT baseline's format: TRL
prompt-completion, GG's prompt and answer. Train: CodeNet's SFT-only problems (every built solution) and GG's released
O2 pairs, in that order of precedence for exact x86 duplicates. No RL problem (D5, Enzo, Nov 10): CodeNet train problems
and MBPP-C train stay out, rows whose x86 equals an RL task's (rl.sources) are left out, and every document with a drop
row in ggrpo.decontam's output (eval or RL item) is left out. So is any row (train or val) whose prompt or answer
is over sft.max_prompt_tokens or sft.max_answer_tokens under the base model's tokenizer. Texts over sft.max_bytes cannot
fit and are left out before tokenizing: the tokenizer panics on two GG documents with 20 MB of spaces. Val, for the held-out SFT loss:
MBPP-C dev and CodeNet val. Rows keep their source and document id (provenance). Two passes: x86 hashes decide which rows are
kept, then the rows are streamed to parquet, so GG's 1.3M pairs never sit in memory at once."""
import polars as pl
import structlog
from huggingface_hub import hf_hub_download
from pydantic_settings import CliApp
from tokenizers import Tokenizer

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
content = pl.col("prompt").list.first().struct.field("content")
tokenizer = Tokenizer.from_file(hf_hub_download(settings.rl.tokenizer, "tokenizer.json",
                                                revision=settings.rl.tokenizer_revision))


def count_tokens(texts: pl.Series) -> pl.Series:
    return pl.Series([len(encoding.ids) for encoding in tokenizer.encode_batch(texts.to_list())], dtype=pl.Int64)


fits = (
    pl.format(settings.prompt.template, "x86").map_batches(count_tokens, return_dtype=pl.Int64, is_elementwise=True)
    <= settings.sft.max_prompt_tokens
) & (
    pl.format(settings.prompt.answer, "arm").map_batches(count_tokens, return_dtype=pl.Int64, is_elementwise=True)
    <= settings.sft.max_answer_tokens
)


def to_conversation(frame: pl.LazyFrame) -> pl.LazyFrame:
    """(source, document_id, x86, arm) -> (source, document_id, prompt, completion), as ggrpo.data writes SFT rows."""
    user = pl.struct(role=pl.lit("user"), content=pl.format(settings.prompt.template, "x86"))
    assistant = pl.struct(role=pl.lit("assistant"), content=pl.format(settings.prompt.answer, "arm"))
    return frame.select("source", "document_id", prompt=pl.concat_list(user), completion=pl.concat_list(assistant))


def read_tasks(path) -> pl.LazyFrame:
    """A task file as (source, document_id, x86, arm)."""
    return pl.scan_ndjson(path).select(
        source=pl.col("task_id").str.extract(r"^([^/]+)/"), document_id="task_id",
        x86=content.str.extract(r"(?s)```x86asm\n(.*)\n```$", 1), arm="arm_ref",
    )


solutions = pl.read_parquet(settings.codenet.solutions).filter(pl.col("split") == "sft_only")
built = pl.read_parquet(settings.codenet.built / "part-*.parquet", columns=["solution_id"])
if missing := solutions.join(built, on="solution_id", how="anti").height:
    raise ValueError(f"{missing} SFT-only CodeNet solutions are not in {settings.codenet.built} yet")
rl_keys = pl.concat([read_tasks(path) for path in settings.rl.sources]).select(key=pl.col("x86").hash()).collect()["key"]
sources = [   # in order of precedence for exact x86 duplicates
    solutions.lazy().join(pl.scan_parquet(settings.codenet.built / "part-*.parquet"), on="solution_id")
    .filter(pl.col("x86").is_not_null() & pl.col("arm").is_not_null())
    .select(source=pl.lit("codenet"), document_id=pl.format("codenet/{}/{}", "problem_id", "solution_id"),
            x86="x86", arm="arm"),
    pl.scan_parquet(settings.decontam.gg.download() / "data").select(
        source=pl.format("gg/{}", "source"), document_id="file", x86="x86", arm="arm"
    ),
]

pairs = pl.read_parquet(settings.decontam.out)
contaminated = pairs.filter("drop")["document_id"].unique()
tokenizable = (pl.col("x86").str.len_bytes() <= settings.sft.max_bytes) & (pl.col("arm").str.len_bytes() <= settings.sft.max_bytes)
keys = pl.concat([frame.filter(tokenizable).select("document_id", key=pl.col("x86").hash(), fits=fits)
                  .collect(engine="streaming") for frame in sources])
if keys["document_id"].is_duplicated().any():
    raise ValueError("document ids must be unique across sources")
kept = keys.unique("key", keep="first", maintain_order=True).filter(
    ~pl.col("document_id").is_in(contaminated.implode()), "fits", ~pl.col("key").is_in(rl_keys.implode())
)

settings.sft.train.parent.mkdir(parents=True, exist_ok=True)
to_conversation(pl.concat(sources).filter(pl.col("document_id").is_in(kept["document_id"].implode()))).sink_parquet(
    settings.sft.train
)
leaked = pl.scan_parquet(settings.sft.train).filter(pl.col("document_id").is_in(contaminated.implode())).select(pl.len())
if leaked.collect().item():
    raise ValueError(f"contaminated documents reached {settings.sft.train}")
val = pl.concat([read_tasks(settings.data.dev), read_tasks(settings.codenet.tasks["val"])]).filter(fits)
to_conversation(val).sink_parquet(settings.sft.val)

train = pl.scan_parquet(settings.sft.train).group_by("source").agg(rows=pl.len()).sort("source").collect()
log.info("wrote", train=str(settings.sft.train), val=str(settings.sft.val), documents=keys.height,
         duplicates=keys.height - keys["key"].n_unique(), contaminated=contaminated.len(),
         too_long=keys.filter(~pl.col("fits")).height, rl_copies=keys.filter(pl.col("key").is_in(rl_keys.implode())).height,
         rows=train.to_dicts(), val_rows=pl.scan_parquet(settings.sft.val).select(pl.len()).collect().item())
