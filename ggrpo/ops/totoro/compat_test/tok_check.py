"""tok_check.py MODEL_DIR BASE_DIR: encode every eval/dev prompt (eval path) and reference, decode references and random
id runs with skip_special_tokens, model-dir tokenizer vs base tokenizer, in this venv's transformers."""
import json
import random
import sys

import transformers
from transformers import AutoTokenizer

tm, tb = AutoTokenizer.from_pretrained(sys.argv[1]), AutoTokenizer.from_pretrained(sys.argv[2])
rows = [json.loads(l) for f in ["eval.jsonl", "dev.jsonl"] for l in open(f"/home/eshiraishi/ggrpo/data/{f}")]
enc = dec = n = 0
for r in rows:
    s = [t.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for t in (tm, tb)]
    ids = [t([x]).input_ids[0] for t, x in zip((tm, tb), s)]
    ref = [t(" " + r["arm_ref"] + "<|im_end|><|endoftext|><|im_start|>").input_ids for t in (tm, tb)]
    enc += (s[0] != s[1]) + (ids[0] != ids[1]) + (ref[0] != ref[1])
    dec += tm.decode(ref[1], skip_special_tokens=True) != tb.decode(ref[1], skip_special_tokens=True)
    n += 1
rng = random.Random(0)
rand = [[rng.randrange(151665) for _ in range(64)] for _ in range(2000)]
dec_rand = sum(tm.decode(x, skip_special_tokens=k) != tb.decode(x, skip_special_tokens=k) for x in rand for k in (True, False))
print(transformers.__version__, f"rows={n} encode_mismatches={enc} decode_ref_mismatches={dec} "
      f"decode_random_mismatches={dec_rand}/4000")
