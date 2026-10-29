import polars as pl
new = pl.read_parquet("data/mbpp_armv8_O2.parquet")
old = pl.read_parquet("data/raw/mbpp_armv8_O2_ventura.parquet").with_columns(
    driver=pl.col("support_src").str.json_decode(pl.Struct({"test.c": pl.String})).struct.field("test.c"))
j = new.join(old, on="task_id", how="full", validate="1:1", suffix="_old")
print("rows", new.height, old.height, j.height)
print(j.select([(pl.col(c) == pl.col(f"{c}_old")).sum().alias(c) for c in ["x86", "arm_ref", "driver", "ref_verdict"]]))
