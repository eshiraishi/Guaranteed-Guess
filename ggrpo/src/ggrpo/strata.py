"""Pass rates by similarity to a model's training data (Mac, --extra data, after ggrpo.decontam), as the decontamination
protocol asks. Each eval item gets one stratum from its pairs with the documents strata.train holds:
- high: some pair reaches a drop threshold;
- middle: some pair reaches decontam.report (0.2), none drops;
- clean: no pair reaches decontam.report, and the asm or C rule could check the item;
- unmeasured: neither rule could check it, so its similarity is unknown.
A model trained without the dropped documents (strata.decontaminated) has an empty high stratum by construction. Results
are the scored files of one model. Each must name known eval items, or the run fails."""
import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")


def read_scored(path) -> pl.DataFrame:
    """A scored file as (item_id, passed). BringUp names its benches; the other sets name tasks."""
    frame = pl.read_ndjson(path)
    item = pl.col("task_id") if "task_id" in frame.columns else pl.format("bringup/{}", "bench")
    return frame.select(item_id=item, passed=pl.col("verdict") == "pass")


source = pl.when(pl.col("document_id").str.starts_with("codenet/")).then(pl.lit("codenet")) \
    .when(pl.col("document_id").str.starts_with("mbpp/")).then(pl.lit("mbpp")).otherwise(pl.lit("gg"))
pairs = pl.read_parquet(settings.decontam.out).filter(source.is_in(settings.strata.train))
if settings.strata.decontaminated:
    pairs = pairs.join(pairs.filter("drop").select("document_id").unique(), on="document_id", how="anti")
strata = pl.read_parquet(settings.decontam.items).join(
    pairs.group_by("item_id").agg(high=pl.col("drop").any()), on="item_id", how="left", validate="1:1"
).with_columns(stratum=pl.when("high").then(pl.lit("high")).when(pl.col("high").is_not_null()).then(pl.lit("middle"))
               .when(pl.col("asm_checked") | pl.col("c_checked")).then(pl.lit("clean")).otherwise(pl.lit("unmeasured")))

results = pl.concat([read_scored(path) for path in settings.strata.scored])
if results["item_id"].is_duplicated().any():
    raise ValueError("strata.scored names an eval item twice")
if unknown := sorted(set(results["item_id"]) - set(strata["item_id"])):
    raise ValueError(f"{len(unknown)} scored items are not eval items: {unknown[:10]}")
joined = results.join(strata, on="item_id", validate="1:1")
table = joined.group_by("set", "stratum").agg(items=pl.len(), passed=pl.col("passed").sum()).with_columns(
    rate=pl.col("passed") / pl.col("items")
).sort("set", "stratum")
settings.strata.out.parent.mkdir(parents=True, exist_ok=True)
joined.select("item_id", "set", "stratum", "passed").write_parquet(settings.strata.out)
for name, group in table.group_by("set", maintain_order=True):
    rates = dict(zip(group["stratum"], group["rate"], strict=True))
    headline_clean = "high" in rates and "clean" in rates and rates["high"] - rates["clean"] > settings.strata.margin
    log.info("stratified", set=name[0], rows=group.drop("set").to_dicts(), headline_clean=headline_clean)
log.info("wrote", path=str(settings.strata.out), items=joined.height, passed=int(joined["passed"].sum()))
