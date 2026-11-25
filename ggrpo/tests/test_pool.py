import polars as pl
import pytest

from ggrpo.pool import select_pool, to_sft
from ggrpo.sample import claim_chunk, count_mixed, get_order, keep_prefix


def census(passes: dict[str, list[bool]]) -> pl.DataFrame:
    return pl.DataFrame([{"rank": rank, "task_id": task, "passed": passed}
                         for rank, (task, outcomes) in enumerate(passes.items()) for passed in outcomes])


def test_order_is_fixed_by_the_seed():
    tasks = pl.DataFrame({"task_id": [f"t{i}" for i in range(50)]})
    first, again, other = get_order(tasks, "a"), get_order(tasks.reverse(), "a"), get_order(tasks, "b")
    assert first["task_id"].to_list() == again["task_id"].to_list()
    assert first["task_id"].to_list() != other["task_id"].to_list()
    assert first["rank"].to_list() == list(range(50))


def test_mixed_tasks_need_every_sample():
    rows = census({"all": [True, True], "none": [False, False], "mixed": [True, False]})
    assert count_mixed(rows, 2) == 1
    with pytest.raises(ValueError, match="exactly 3"):
        count_mixed(rows, 3)


def test_pool_takes_the_first_mixed_tasks_in_census_order():
    rows = census({"a": [True, False], "b": [True, True], "c": [False, True], "d": [False, False], "e": [True, False]})
    assert select_pool(rows, 2)["task_id"].to_list() == ["a", "c"]
    with pytest.raises(ValueError, match="fewer than the cap"):
        select_pool(rows, 4)


def test_pool_ignores_tasks_after_an_unscored_rank():
    rows = census({"a": [True, False], "b": [True, True], "c": [False, True]}).filter(pl.col("task_id") != "b")
    assert keep_prefix(rows)["task_id"].unique().to_list() == ["a"]
    with pytest.raises(ValueError, match="fewer than the cap"):
        select_pool(rows, 2)


def test_shards_claim_disjoint_chunks_lowest_rank_first(tmp_path):
    tasks = pl.DataFrame({"task_id": [f"t{i}" for i in range(5)]}).with_row_index("rank")
    out = tmp_path / "census.jsonl"
    assert claim_chunk(tasks, out, "gpu0", 2)["task_id"].to_list() == ["t0", "t1"]
    assert claim_chunk(tasks, out, "gpu1", 2)["task_id"].to_list() == ["t2", "t3"]
    out.write_text('{"rank": 0, "task_id": "t0"}\n{"rank": 1, "task_id": "t1"}\n')
    assert claim_chunk(tasks, out, "gpu0", 2)["task_id"].to_list() == ["t4"]
    assert claim_chunk(tasks, out, "gpu1", 2)["task_id"].to_list() == ["t2", "t3"]   # a restart takes its claim back


def test_sft_rows_have_the_assistant_turn():
    rows = to_sft(pl.DataFrame({"prompt": [[{"role": "user", "content": "x86"}]], "arm": ["arm"]}), pl.col("arm"))
    assert rows.to_dicts() == [{"prompt": [{"role": "user", "content": "x86"}],
                                "completion": [{"role": "assistant", "content": "arm"}]}]
