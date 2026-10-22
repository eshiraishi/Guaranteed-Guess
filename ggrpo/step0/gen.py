"""Beam-search generation with ahmedheakl/gg-armv8-O2, paper settings (chat template, exact prompt,
no sampling). Usage: gen.py tasks.jsonl out.jsonl --device mps|cuda|cpu [--beams 8] [--ids humaneval/3,...] [--limit N]
Output JSONL: {"task_id","pred","raw","prompt_tokens","new_tokens","seconds"}; resumes by skipping done task_ids."""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--model", default="ahmedheakl/gg-armv8-O2")
ap.add_argument("--device", default="mps"); ap.add_argument("--beams", type=int, default=8)
ap.add_argument("--max_new_tokens", type=int, default=16000)
ap.add_argument("--ids", default=""); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
tasks = [json.loads(l) for l in open(a.tasks)]
if a.ids: tasks = [t for t in tasks if t["task_id"] in set(a.ids.split(","))]
if a.limit: tasks = tasks[:a.limit]
done = set(json.loads(l)["task_id"] for l in open(a.out)) if os.path.exists(a.out) else set()
with open(a.out, "a") as f:
    for t in tasks:
        if t["task_id"] in done: continue
        prompt = f"Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{t['x86']}\n```"
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt").to(a.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=a.beams)
        new = out[0][enc.input_ids.shape[1]:]
        raw = tok.decode(new, skip_special_tokens=True)
        pred = raw.split("```armasm\n")[-1].split("```")[0]
        f.write(json.dumps({"task_id": t["task_id"], "pred": pred, "raw": raw, "prompt_tokens": int(enc.input_ids.shape[1]),
                            "new_tokens": int(new.shape[0]), "seconds": round(time.time() - t0, 1)}) + "\n"); f.flush()
        print(t["task_id"], enc.input_ids.shape[1], int(new.shape[0]), round(time.time() - t0, 1), flush=True)
