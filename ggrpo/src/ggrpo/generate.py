"""Generate ARM for each eval task with HF generate as in eval/eval_huamaneval.py; appends to generate.out, resumable."""
import json
import time

import polars as pl
import structlog
import torch
from pydantic_settings import CliApp
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
set_seed(settings.generate.seed)
tokenizer = AutoTokenizer.from_pretrained(settings.generate.model, revision=settings.generate.revision)
model = AutoModelForCausalLM.from_pretrained(
    settings.generate.model,
    revision=settings.generate.revision,
    dtype=torch.bfloat16,
    attn_implementation=settings.generate.attn_implementation,
    device_map=settings.generate.device,
)
# GG uses rope_theta 1e6. Checkpoints saved by transformers >= 5 store it under rope_parameters, which 4.x silently
# ignores (theta 1e4, wrong logits, no warning): fail loudly on any loader that gets it wrong.
inv_freq = float(model.model.rotary_emb.inv_freq[1])
assert abs(inv_freq - 1e6 ** (-2 / 128)) < 5e-3, f"rope_theta is not 1e6 (inv_freq[1] = {inv_freq})"

settings.generate.out.parent.mkdir(parents=True, exist_ok=True)
out = settings.generate.out
# a crash before the first row leaves an empty file, which polars can't read
done = pl.read_ndjson(out)["task_id"] if out.exists() and out.stat().st_size else pl.Series(dtype=pl.String)
with settings.generate.out.open("a") as f:
    for task in pl.read_ndjson(settings.data.eval).filter(~pl.col("task_id").is_in(done.implode())).to_dicts():
        inputs = tokenizer(
            [tokenizer.apply_chat_template(task["prompt"], tokenize=False, add_generation_prompt=True)],
            return_tensors="pt",
        ).to(model.device)
        n_prompt = inputs.input_ids.shape[1]
        start = time.monotonic()
        new = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=GenerationConfig.from_dict(
                getattr(settings.generate, settings.generate.mode).model_dump()
                | {"max_new_tokens": settings.generate.max_total_tokens - n_prompt}
            ),
        )[0, n_prompt:]
        record = {"task_id": task["task_id"], "prompt_tokens": n_prompt, "new_tokens": len(new),
                  "seconds": round(time.monotonic() - start, 1)}
        f.write(json.dumps(record | {"raw": tokenizer.decode(new, skip_special_tokens=True)}) + "\n")
        f.flush()
        log.info("generated", mode=settings.generate.mode, record=record)
