import sys, torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.bfloat16)
x = float(m.model.rotary_emb.inv_freq[1]); ok = abs(x - 1e6 ** (-2 / 128)) < 5e-3
print(sys.argv[1].split("/")[-1][:30], m.model.rotary_emb.inv_freq.dtype, round(x, 5), "PASS" if ok else "FAIL")
