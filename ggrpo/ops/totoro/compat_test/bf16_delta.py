"""Fraction of elements changed vs base: fp32 checkpoint, and after the bf16 cast the evals do; plus revision= with a local dir."""
import sys, torch, transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
ck, base = sys.argv[1:3]
c = AutoConfig.from_pretrained(ck, revision="0444a378f384300187c7f43914fc986a81ff051a")
AutoTokenizer.from_pretrained(ck, revision="0444a378f384300187c7f43914fc986a81ff051a")
print(transformers.__version__, "local dir + revision= loads; rope_theta", getattr(c, "rope_theta", None))
n = ch32 = ch16 = 0
with safe_open(ck + "/model.safetensors", "pt") as f, safe_open(base + "/model.safetensors", "pt") as g:
    for k in f.keys():
        p, b = f.get_tensor(k), g.get_tensor(k)
        n += p.numel(); ch32 += (p != b.float()).sum().item(); ch16 += (p.to(torch.bfloat16) != b).sum().item()
print(f"elements={n} changed_fp32={ch32/n:.4%} changed_after_bf16_cast={ch16/n:.4%}")
