"""BringUp O2 generation with ahmedheakl/gg-armv8-O2 (prompt and decode as eval/eval_bringupbench.py: chat template,
paper prompt, skip_special_tokens, bf16, device_map auto, flash_attention_2).
--mode released : eval/eval_bringupbench.py verbatim: generate(max_new_tokens=32768, do_sample=False) -> greedy.
--mode beam8    : paper text (pass@1, beam 8, no sampling) with eval_huamaneval.py's cap max_new_tokens = 32000 - prompt;
                  prompts with >= 32000 tokens are recorded as error="context_overflow" (no generation).
Output JSONL (resumable): file, mode, prompt_tokens, new_tokens, seconds, raw, error, versions."""
import argparse, json, os, time
import pyarrow.parquet as pq
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--mode", choices=["released", "beam8"], required=True)
ap.add_argument("--model", default="ahmedheakl/gg-armv8-O2"); ap.add_argument("--revision", default="0444a378f384300187c7f43914fc986a81ff051a")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gpu", default="RTX PRO 6000")   # one GPU type per run: refuse any other
ap.add_argument("--files", default="")   # comma-separated subset (one Slurm job per row)
a = ap.parse_args()
gpu = torch.cuda.get_device_name(0)
assert a.gpu in gpu, gpu
rows = [r for r in pq.read_table(a.data).to_pylist() if not a.files or r["file"] in a.files.split(",")]
rows = sorted(rows, key=lambda r: len(r["x86"]))[: a.limit or None]   # short first: early, useful partial results
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
inv_freq = float(model.model.rotary_emb.inv_freq[1])   # rope_theta must be 1e6: transformers 5 checkpoints store it under
assert abs(inv_freq - 1e6 ** (-2 / 128)) < 5e-3, inv_freq   # rope_parameters, which 4.50 silently ignores (theta 1e4)
done = {json.loads(l)["file"] for l in open(a.out)} if os.path.exists(a.out) else set()
with open(a.out, "a") as f:
    for ex in rows:
        if ex["file"] in done: continue
        prompt = f"Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{ex['x86']}\n```"
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([tokens], return_tensors="pt").to(model.device)
        n = inputs.input_ids.shape[1]
        t0, error, new = time.time(), None, []
        torch.cuda.reset_peak_memory_stats()
        if a.mode == "beam8" and n >= 32000:
            error = "context_overflow"
        else:
            kw = dict(max_new_tokens=32768, do_sample=False) if a.mode == "released" else \
                 dict(max_new_tokens=32000 - n, num_beams=8, num_return_sequences=1, do_sample=False)
            try:
                new = model.generate(**inputs, **kw)[0][n:]
            except torch.cuda.OutOfMemoryError:   # recorded, never silently retried
                error = "oom"; torch.cuda.empty_cache()
        raw = tokenizer.batch_decode([new], skip_special_tokens=True)[0] if len(new) else ""
        rec = {"file": ex["file"], "mode": a.mode, "prompt_tokens": n, "new_tokens": len(new), "seconds": round(time.time() - t0, 1),
               "raw": raw, "error": error, "gpu": gpu, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(ex["file"], n, len(new), rec["seconds"], error, flush=True)
