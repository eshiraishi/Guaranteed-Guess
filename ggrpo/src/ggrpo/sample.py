"""Sample sample.samples completions per task with vLLM at GRPO's sampling settings, score them through the reward server
and append them to sample.out, a chunk of tasks at a time, in a fixed random order (sha256(sample.seed:task_id)).
Resumable: tasks already in sample.out are skipped. Training host GPU1 or an eval GPU, in the train venv.
- The census (prereg.md §2; config.toml's defaults): the RL tasks from the fp16 RL start; stops once sample.stop_mixed
  tasks have 0 < passes < samples. ggrpo.pool then builds the pool from these rows.
- Validation E2 of a checkpoint: --sample.tasks data/codenet/val.jsonl --sample.limit 100 (the fixed 100-task subset is
  the first 100 in this order) --sample.stop_mixed null --sample.model <checkpoint> --sample.out <file>.
Never the test split: it is scored once, at the end, after the differential tests are frozen.
Rows: rank (position in the order), task_id, sample, new_tokens, finish_reason, verdict, tests, passed, raw."""
import hashlib
import json

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


def main() -> None:
    from vllm import LLM, SamplingParams   # the train venv only

    from ggrpo.reward import post_scores

    settings = CliApp.run(Settings)
    config = settings.sample
    if config.tasks.resolve() == settings.codenet.tasks["test"].resolve():
        raise SystemExit("the test split is scored once, at the end, after the differential tests are frozen")
    tasks = get_order(pl.read_ndjson(config.tasks).select("task_id", "prompt"), config.seed)
    if config.limit is not None:
        tasks = tasks.head(config.limit)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    done = pl.read_ndjson(config.out) if config.out.exists() and config.out.stat().st_size else None
    if done is not None:
        tasks = tasks.filter(~pl.col("task_id").is_in(done["task_id"].implode()))
    llm = LLM(model=str(config.model), dtype=config.dtype, gpu_memory_utilization=config.gpu_memory_utilization,
              max_model_len=config.max_model_len, seed=0)
    rope = llm.llm_engine.model_config.hf_config.rope_parameters["rope_theta"]
    assert rope == 1e6, rope   # see generate.py: a silently wrong rope_theta corrupts every output
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(n=config.samples, temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                            repetition_penalty=1.0, max_tokens=config.max_tokens)
    for chunk in tasks.iter_slices(config.chunk):
        prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                   for prompt in chunk["prompt"].to_list()]
        rows = [
            {"rank": task["rank"], "task_id": task["task_id"], "sample": number, "new_tokens": len(output.token_ids),
             "finish_reason": output.finish_reason, "raw": output.text}
            for task, result in zip(chunk.to_dicts(), llm.generate(prompts, params), strict=True)
            for number, output in enumerate(result.outputs)
        ]
        results = [result for start in range(0, len(rows), config.score_batch)
                   for result in post_scores([{"task_id": row["task_id"], "completion": row["raw"]}
                                              for row in rows[start:start + config.score_batch]])]
        with config.out.open("a") as f:
            for row, result in zip(rows, results, strict=True):
                f.write(json.dumps(row | {key: result[key] for key in ("verdict", "tests", "passed")}) + "\n")
        scored = pl.read_ndjson(config.out)
        mixed = count_mixed(scored, config.samples)
        log.info("sampled", tasks=scored["task_id"].n_unique(), mixed=mixed, pass_rate=round(scored["passed"].mean(), 4))
        if config.stop_mixed is not None and mixed >= config.stop_mixed:
            log.info("stopped", reason="stop_mixed reached", mixed=mixed)
            break


if __name__ == "__main__":
    main()
