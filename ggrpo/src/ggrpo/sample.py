"""Sample sample.samples completions per task with vLLM at GRPO's sampling settings, score them through the reward server
and append them to sample.out, a chunk of tasks at a time, in a fixed random order (sha256(sample.seed:task_id)).
Resumable: tasks already in sample.out are skipped. Training host GPU1 or an eval GPU, in the train venv.
Shards: processes on one host with the same sample.out and different sample.shard split the work. Under a lock on
sample.out, each claims the lowest-ranked chunk that no one has scored or claimed (its claim is sample.out's
.claim.<shard> file), and appends its rows when scored. Only the scored prefix of the order counts (tasks before the
first unscored rank), so the result is what one process would have written. A restarted shard (same name) takes its
unscored claim back; a shard that is never restarted stalls the prefix, and the others then sample past the stop.
- The census (prereg.md §2; config.toml's defaults): the RL tasks from the fp16 RL start; stops once sample.stop_mixed
  tasks have 0 < passes < samples. ggrpo.pool then builds the pool from these rows.
- Validation E2 of a checkpoint: --sample.tasks data/codenet/val.jsonl --sample.limit 100 (the fixed 100-task subset is
  the first 100 in this order) --sample.stop_mixed null --sample.model <checkpoint> --sample.out <file>.
Never the test split: it is scored once, at the end, after the differential tests are frozen.
Rows: rank (position in the order), task_id, sample, new_tokens, finish_reason, verdict, tests, passed, raw."""
import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

log = structlog.get_logger("ggrpo")


def get_order(tasks: pl.DataFrame, seed: str) -> pl.DataFrame:
    """The tasks in the fixed random order, with their rank in it."""
    key = pl.col("task_id").map_elements(lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
                                         return_dtype=pl.String)
    return tasks.sort(key).with_row_index("rank")


def count_mixed(rows: pl.DataFrame, samples: int) -> int:
    """Tasks whose samples disagree (0 < passes < samples), counting only tasks with every sample scored."""
    per_task = rows.group_by("task_id").agg(n=pl.len(), passes=pl.col("passed").sum())
    if (incomplete := per_task.filter(pl.col("n") != samples)).height:
        raise ValueError(f"tasks without exactly {samples} samples: {incomplete['task_id'].to_list()[:5]}")
    return per_task.filter((pl.col("passes") > 0) & (pl.col("passes") < samples)).height


def keep_prefix(rows: pl.DataFrame) -> pl.DataFrame:
    """The rows of the scored prefix of the order: tasks ranked before the first rank with no rows."""
    ranks = set(rows["rank"].to_list())
    end = next(rank for rank in range(len(ranks) + 1) if rank not in ranks)
    return rows.filter(pl.col("rank") < end)


@contextmanager
def lock_out(out: Path) -> Iterator[None]:
    with out.with_name(out.name + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_rows(out: Path) -> pl.DataFrame | None:
    return pl.read_ndjson(out) if out.exists() and out.stat().st_size else None


def claim_chunk(tasks: pl.DataFrame, out: Path, shard: str, size: int) -> pl.DataFrame:
    """Under lock_out: the lowest-ranked size tasks neither scored nor claimed by another shard, as this shard's claim."""
    mine = out.with_name(f"{out.name}.claim.{shard}")
    taken = [task_id for path in out.parent.glob(f"{out.name}.claim.*") if path != mine
             for task_id in json.loads(path.read_text())]
    if (scored := read_rows(out)) is not None:
        taken += scored["task_id"].unique().to_list()
    chunk = tasks.filter(~pl.col("task_id").is_in(taken)).head(size)
    mine.write_text(json.dumps(chunk["task_id"].to_list()))
    return chunk


def main() -> None:
    from vllm import LLM, SamplingParams   # the train venv only

    from ggrpo.reward import post_scores

    settings = CliApp.run(Settings)
    config = settings.sample
    if not config.score and config.stop_mixed is not None:
        raise SystemExit("stop_mixed needs the scores: unset it or score while sampling")
    if config.tasks.resolve() == settings.codenet.tasks["test"].resolve():
        raise SystemExit("the test split is scored once, at the end, after the differential tests are frozen")
    tasks = get_order(pl.read_ndjson(config.tasks).select("task_id", "prompt"), config.seed)
    if config.limit is not None:
        tasks = tasks.head(config.limit)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=str(config.model), dtype=config.dtype, gpu_memory_utilization=config.gpu_memory_utilization,
              max_model_len=config.max_model_len, seed=0)
    rope = llm.llm_engine.model_config.hf_config.rope_parameters["rope_theta"]
    assert rope == 1e6, rope   # see generate.py: a silently wrong rope_theta corrupts every output
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(n=config.samples, temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                            repetition_penalty=1.0, max_tokens=config.max_tokens)
    while True:
        with lock_out(config.out):
            chunk = claim_chunk(tasks, config.out, config.shard, config.chunk)
        if not chunk.height:
            log.info("stopped", reason="no task left to claim")
            break
        prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                   for prompt in chunk["prompt"].to_list()]
        rows = [
            {"rank": task["rank"], "task_id": task["task_id"], "sample": number, "new_tokens": len(output.token_ids),
             "finish_reason": output.finish_reason, "raw": output.text}
            # use_tqdm=False, not TQDM_DISABLE: with tqdm disabled vLLM still builds a bar and then divides
            # total_in_toks by its elapsed time, which is 0 (offline_utils.py:614). That killed all three E2
            # LR-grid runs on 2026-11-20 at 03:18-03:21Z, after 58 s each, with no output written.
            for task, result in zip(chunk.to_dicts(), llm.generate(prompts, params, use_tqdm=False),
                                    strict=True)
            for number, output in enumerate(result.outputs)
        ]
        if not config.score:
            with lock_out(config.out):
                with config.out.open("a") as f:
                    f.writelines(json.dumps(row) + "\n" for row in rows)
            log.info("sampled", shard=config.shard, rows=len(rows))
            continue
        results = [result for start in range(0, len(rows), config.score_batch)
                   for result in post_scores([{"task_id": row["task_id"], "completion": row["raw"]}
                                              for row in rows[start:start + config.score_batch]])]
        with lock_out(config.out):
            with config.out.open("a") as f:
                f.writelines(json.dumps(row | {key: result[key] for key in ("verdict", "tests", "passed")}) + "\n"
                             for row, result in zip(rows, results, strict=True))
            scored = keep_prefix(pl.read_ndjson(config.out))
        mixed = count_mixed(scored, config.samples)
        log.info("sampled", shard=config.shard, prefix_tasks=scored["task_id"].n_unique(), mixed=mixed,
                 # the prefix is empty while another shard still holds rank 0 (Nov 20, 20:39Z: round(None) killed shard 1)
                 pass_rate=round(scored["passed"].mean(), 4) if scored.height else None)
        if config.stop_mixed is not None and mixed >= config.stop_mixed:
            log.info("stopped", reason="stop_mixed reached", mixed=mixed)
            break


if __name__ == "__main__":
    main()
