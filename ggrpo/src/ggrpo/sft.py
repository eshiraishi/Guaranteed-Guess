"""SFT train and val files (Mac, --extra data, after ggrpo.decontam) in the SFT baseline's format: TRL
prompt-completion, GG's prompt and answer. Train: MBPP-C train, the SFT side of CodeNet (every built solution of
SFT-only and train problems) and GG's released O2 pairs, in that order of precedence for exact x86 duplicates. Every
document with a drop row in ggrpo.decontam's output is left out. Val, for the held-out SFT loss: MBPP-C dev
and CodeNet val. Rows keep their source and document id (provenance). Two passes: x86 hashes decide which rows are
kept, then the rows are streamed to parquet, so GG's 1.3M pairs never sit in memory at once."""
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
content = pl.col("prompt").list.first().struct.field("content")


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


solutions = pl.read_parquet(settings.codenet.solutions).filter(pl.col("split").is_in(["sft_only", "train"]))
built = pl.read_parquet(settings.codenet.built / "part-*.parquet", columns=["solution_id"])
if missing := solutions.join(built, on="solution_id", how="anti").height:
    raise ValueError(f"{missing} SFT-side CodeNet solutions are not in {settings.codenet.built} yet")
sources = [   # in order of precedence for exact x86 duplicates
    read_tasks(settings.data.train).filter(~pl.col("document_id").is_in(settings.data.sft_exclude)),
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
keys = pl.concat([frame.select("document_id", key=pl.col("x86").hash()).collect() for frame in sources])
if keys["document_id"].is_duplicated().any():
    raise ValueError("document ids must be unique across sources")
kept = keys.unique("key", keep="first", maintain_order=True).filter(
    ~pl.col("document_id").is_in(contaminated.implode())
)

settings.sft.train.parent.mkdir(parents=True, exist_ok=True)
to_conversation(pl.concat(sources).filter(pl.col("document_id").is_in(kept["document_id"].implode()))).sink_parquet(
    settings.sft.train
)
leaked = pl.scan_parquet(settings.sft.train).filter(pl.col("document_id").is_in(contaminated.implode())).select(pl.len())
if leaked.collect().item():
    raise ValueError(f"contaminated documents reached {settings.sft.train}")
val = pl.concat([read_tasks(settings.data.dev), read_tasks(settings.codenet.tasks["val"])])
to_conversation(val).sink_parquet(settings.sft.val)

train = pl.scan_parquet(settings.sft.train).group_by("source").agg(rows=pl.len()).sort("source").collect()
log.info("wrote", train=str(settings.sft.train), val=str(settings.sft.val), documents=keys.height,
         duplicates=keys.height - keys["key"].n_unique(), contaminated=contaminated.len(),
         rows=train.to_dicts(), val_rows=pl.scan_parquet(settings.sft.val).select(pl.len()).collect().item())
