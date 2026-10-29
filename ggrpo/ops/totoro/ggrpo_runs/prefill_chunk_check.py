"""Does transformers' chunked prefill (prefill_chunk_size=512) change generation? bf16 and NF4 bases.

Batch of 8 training prompts, left-padded as TRL does, including the longest one (10,749 tokens).
Compares the first-token logits and 256 greedy tokens with and without chunking, and records peak memory.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

REPO, REV = "ahmedheakl/gg-armv8-O2", "0444a378f384300187c7f43914fc986a81ff051a"
tok = AutoTokenizer.from_pretrained(REPO, revision=REV, padding_side="left")
rows = [json.loads(line) for line in open("data/train.jsonl")]
ids = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False) for r in rows]
order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
picked = [order[-1]] + order[:: len(order) // 7][:7]  # longest + spread over the length distribution
batch = tok.pad({"input_ids": [ids[i] for i in picked]}, return_tensors="pt").to("cuda")
nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

for arm, quant in [("bf16", None), ("nf4", nf4)]:
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": 0})
    out = {}
    for chunk in [None, 512]:
        config = GenerationConfig(max_new_tokens=256, do_sample=False, prefill_chunk_size=chunk, disable_compile=True,
                                  pad_token_id=tok.pad_token_id, return_dict_in_generate=True, output_logits=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            gen = model.generate(**batch, generation_config=config)
        out[chunk] = (gen.sequences[:, batch["input_ids"].shape[1]:], gen.logits[0].float(),
                      torch.cuda.max_memory_allocated() / 2**30)
    (a, la, ma), (b, lb, mb) = out[None], out[512]
    first_diff = [int((x != y).nonzero()[0]) if (x != y).any() else None for x, y in zip(a, b)]
    print(json.dumps({"arm": arm, "prompt_lengths": [len(ids[i]) for i in picked],
                      "first_token_logit_max_abs_diff": (la - lb).abs().max().item(),
                      "first_token_argmax_equal": bool((la.argmax(-1) == lb.argmax(-1)).all()),
                      "greedy_tokens_equal": (a == b).float().mean().item(), "first_divergence": first_diff,
                      "peak_gib_full_prefill": round(ma, 1), "peak_gib_chunked": round(mb, 1)}))
    del model
    torch.cuda.empty_cache()
