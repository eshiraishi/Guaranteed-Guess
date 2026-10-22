"""Teacher-forced log-probs of fixed completions (paper prompt + completion), to compare package environments and
attention kernels on the same GPU. Runs under transformers 4.50 (extra paper) and 5.x (extra train).
When equivalence.reference exists and differs from equivalence.out, logs how far the two files agree."""
import hashlib
import json

import polars as pl
import structlog
import torch
from pydantic_settings import CliApp
from transformers import AutoModelForCausalLM, AutoTokenizer

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
tokenizer = AutoTokenizer.from_pretrained(settings.generate.model, revision=settings.generate.revision)
model = AutoModelForCausalLM.from_pretrained(
    settings.generate.model,
    revision=settings.generate.revision,
    torch_dtype=torch.bfloat16,   # `dtype` in transformers 5; `torch_dtype` is kept there for compatibility
    attn_implementation=settings.equivalence.attn_implementation,
    device_map=settings.generate.device,
)

settings.equivalence.out.parent.mkdir(parents=True, exist_ok=True)
skipped = []
with settings.equivalence.out.open("w") as f:
    for task in (
        pl.read_ndjson(settings.data.eval)
        .join(pl.read_ndjson(settings.equivalence.completions).select("task_id", "raw"), on="task_id", validate="1:1")
        .to_dicts()
    ):
        # the same two-step tokenization as eval/eval_huamaneval.py
        prompt_ids = tokenizer(
            [tokenizer.apply_chat_template(task["prompt"], tokenize=False, add_generation_prompt=True)]
        ).input_ids[0]
        completion_ids = tokenizer([task["raw"]], add_special_tokens=False).input_ids[0]
        if len(prompt_ids) + len(completion_ids) > settings.equivalence.max_tokens:
            skipped.append(task["task_id"])
            continue
        ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt_ids) - 1 : -1].float()
        f.write(json.dumps({
            "task_id": task["task_id"],
            "prompt_ids": hashlib.sha256(str(prompt_ids).encode()).hexdigest(),
            "completion_ids": hashlib.sha256(str(completion_ids).encode()).hexdigest(),
            "logprobs": logits.log_softmax(-1).gather(-1, ids[0, len(prompt_ids):, None])[:, 0].tolist(),
            "argmax": logits.argmax(-1).tolist(),
        }) + "\n")
log.info("equivalence.wrote", out=str(settings.equivalence.out), skipped=skipped,
         attn_implementation=settings.equivalence.attn_implementation)

if settings.equivalence.reference.exists() and settings.equivalence.reference != settings.equivalence.out:
    pair = pl.read_ndjson(settings.equivalence.out).join(
        pl.read_ndjson(settings.equivalence.reference), on="task_id", suffix="_ref", validate="1:1"
    )
    same_ids = pair.filter(
        pl.col("prompt_ids") == pl.col("prompt_ids_ref"), pl.col("completion_ids") == pl.col("completion_ids_ref")
    )
    log.info(
        "equivalence.compared",
        out=str(settings.equivalence.out),
        reference=str(settings.equivalence.reference),
        tasks=pair.height,
        same_token_ids=same_ids.height,
        sequences=same_ids.select(
            max_abs_logprob_sum_diff=(pl.col("logprobs").list.sum() - pl.col("logprobs_ref").list.sum()).abs().max(),
        ).row(0, named=True),
        tokens=same_ids.explode("logprobs", "logprobs_ref", "argmax", "argmax_ref").select(
            tokens=pl.len(),
            max_abs_logprob_diff=(pl.col("logprobs") - pl.col("logprobs_ref")).abs().max(),
            mean_abs_logprob_diff=(pl.col("logprobs") - pl.col("logprobs_ref")).abs().mean(),
            argmax_agreement=(pl.col("argmax") == pl.col("argmax_ref")).mean(),
        ).row(0, named=True),
    )
