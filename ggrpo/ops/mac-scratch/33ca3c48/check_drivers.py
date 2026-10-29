import polars as pl
from ggrpo.build import find_definitions
shared = (find_definitions(pl.concat_str("func_dep", "func"))
    .list.set_intersection(find_definitions(pl.concat_str("func_dep", "test")))
    .list.set_difference(pl.lit(["main", "func0"])).list.sort())
prototype = pl.col("func").str.extract(r"(?:^|\n)([^\n;{}#]*\bfunc0\s*\([^{]*)\{", 1).str.strip_chars() + ";"
new = (pl.read_parquet("data/raw/decompile_eval_mbpp.parquet").sort("index").with_row_index("rank")
    .select(task_id=pl.format("mbpp/{}", "rank"), driver=pl.concat_str(
        shared.list.eval(pl.format("#pragma weak {}\n", pl.element())).list.join(""),
        "func_dep", pl.lit("\n"), prototype, pl.lit("\n"), "test")).drop_nulls("driver"))
old = pl.read_parquet("data/raw/mbpp_armv8_O2_ventura.parquet").select(
    "task_id", old=pl.col("support_src").str.json_decode(pl.Struct({"test.c": pl.String})).struct.field("test.c"))
j = new.join(old, on="task_id", how="full", validate="1:1")
print("new", new.height, "old", old.height, "equal", (j["driver"] == j["old"]).sum(), "only_new", j.filter(pl.col("old").is_null())["task_id"].to_list(), "only_old", j.filter(pl.col("driver").is_null())["task_id"].to_list())
