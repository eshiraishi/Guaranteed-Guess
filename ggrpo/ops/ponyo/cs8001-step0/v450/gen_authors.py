"""eval/eval_huamaneval.py's inference(), verbatim apart from I/O: bf16, device_map auto, flash_attention_2,
chat template, max_new_tokens = 32000 - prompt tokens, num_beams=8, num_return_sequences=1.
--mode authors: exactly the authors' call (decoding inherited from generation_config.json -> beam sampling).
--mode greedy_beam: the same call plus do_sample=False (the paper text).
Seeds: the authors set none; here set_seed(seed * 1000 + row) before each prompt so runs are resumable/reproducible.
Usage: gen_authors.py data.parquet out.jsonl --mode authors|greedy_beam [--seed 0] [--limit N]"""
import argparse, json, os, time
import pyarrow.parquet as pq
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--mode", choices=["authors", "greedy_beam"], required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="ahmedheakl/gg-armv8-O2"); ap.add_argument("--revision", default="0444a378f384300187c7f43914fc986a81ff051a")
a = ap.parse_args()
rows = pq.read_table(a.data).to_pylist()[: a.limit or None]
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
done = {json.loads(l)["file"] for l in open(a.out)} if os.path.exists(a.out) else set()
extra = {"do_sample": False} if a.mode == "greedy_beam" else {}
with open(a.out, "a") as f:
    for i, ex in enumerate(rows):
        if ex["file"] in done: continue
        messages = [{"role": "user", "content": f"""Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{ex['x86']}\n```"""}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        max_new_tokens = 32_000 - len(model_inputs["input_ids"][0])
        set_seed(a.seed * 1000 + i)
        t0 = time.time()
        try:
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, num_beams=8, num_return_sequences=1, **extra)
            new, error = generated_ids[0][len(model_inputs.input_ids[0]):], None
        except torch.cuda.OutOfMemoryError:   # recorded, never silently retried
            new, error = [], "oom"
            torch.cuda.empty_cache()
        raw = tokenizer.batch_decode([new], skip_special_tokens=True)[0] if error is None else ""
        rec = {"error": error, "file": ex["file"], "mode": a.mode, "seed": a.seed, "prompt_tokens": len(model_inputs.input_ids[0]),
               "new_tokens": len(new), "seconds": round(time.time() - t0, 1), "raw": raw,
               "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(ex["file"], rec["prompt_tokens"], rec["new_tokens"], rec["seconds"], flush=True)
