"""The extraction regex in config.toml must reproduce the paper's split (eval/eval_huamaneval.py) exactly."""
import polars as pl

from ggrpo.settings import Settings

settings = Settings()


def test_extract_matches_paper_split():
    raws = [
        "",
        "mov x0, #1",
        "```armasm\nmov x0, #1\n```",
        "```armasm\nunclosed fence",
        "a```armasm\nfirst\n```b```armasm\nsecond\n```c",
        "text\n```",
        "```armasm\n```armasm\n",
        "```armasm\nx\n```\n",
        *pl.read_ndjson(settings.data.beam8_cap16k)["raw"].to_list(),
    ]
    assert pl.Series(raws).str.extract(settings.prompt.extract, 1).to_list() == [
        raw.split("```armasm\n")[-1].split("```")[0] for raw in raws
    ]
