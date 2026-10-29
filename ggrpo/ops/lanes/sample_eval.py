"""Secondary dev metric: sampled pass@1 under the training distribution (grpo.yaml: temperature 1.0, no top-k/top-p,
repetition penalty 1.0, 4096 new tokens), 8 samples per task with vLLM. The headline metric stays beam 8 (generate.py);
this one separates "RL improves what it samples" from "RL generalizes to the beam-search eval".
Usage: python sample_eval.py MODEL_DIR_OR_ID TASKS.jsonl OUT.jsonl   (writes task_id, sample, raw)"""
import json
import sys

import polars as pl
from vllm import LLM, SamplingParams

model, tasks, out = sys.argv[1:4]
llm = LLM(model=model, revision="0444a378f384300187c7f43914fc986a81ff051a" if model.startswith("ahmedheakl/") else None,
          dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=16384, seed=0)
rope = llm.llm_engine.model_config.hf_config.rope_parameters["rope_theta"]
assert rope == 1e6, rope   # see generate.py: a silently wrong rope_theta corrupts every output
rows = pl.read_ndjson(tasks).to_dicts()
tokenizer = llm.get_tokenizer()
prompts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
params = SamplingParams(n=8, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=4096, seed=0)
with open(out, "w") as f:
    for r, result in zip(rows, llm.generate(prompts, params)):
        for i, o in enumerate(result.outputs):
            f.write(json.dumps({"task_id": r["task_id"], "sample": i, "new_tokens": len(o.token_ids), "raw": o.text}) + "\n")
