"""How much of a LoRA delta survives TRL's QLoRA sync path (PEFT merge into NF4 = dequantize, add, requantize)?

Uses the real smoke adapter (3 GRPO steps) and the real GG weights. `scale` multiplies the delta to mimic later steps.
Also measures base drift after merge + unmerge (what TRL does every sync), and how often the block absmax changes
(vLLM keeps the absmax from its own load; TRL pushes only the packed codes).
"""

import json
import sys

import bitsandbytes.functional as F
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO, REV = "ahmedheakl/gg-armv8-O2", "0444a378f384300187c7f43914fc986a81ff051a"
adapter = safe_open(sys.argv[1], "pt", device="cuda")
weights = hf_hub_download(REPO, "model.safetensors", revision=REV)


def quantize(w):
    return F.quantize_4bit(w, quant_type="nf4", compress_statistics=True, blocksize=64)


def dequantize(q):
    return F.dequantize_4bit(q[0], q[1]).to(torch.bfloat16)


for name in ["model.layers.0.self_attn.q_proj", "model.layers.14.mlp.down_proj", "model.layers.27.self_attn.v_proj"]:
    with safe_open(weights, "pt", device="cuda") as f:
        w = f.get_tensor(name + ".weight").to(torch.bfloat16)
    a = adapter.get_tensor(f"base_model.model.{name}.lora_A.weight").float()
    b = adapter.get_tensor(f"base_model.model.{name}.lora_B.weight").float()
    delta0 = (b @ a * 32 / 1).to(torch.bfloat16)  # alpha / r
    q = quantize(w)
    base = dequantize(q)
    for scale in [1, 10, 100]:
        delta = delta0 * scale
        merged_q = quantize(base + delta)
        merged = dequantize(merged_q)
        kept = ((merged - base).float() * delta.float()).sum() / (delta.float() ** 2).sum()  # projection of applied update on delta
        err = (merged - base - delta).float().norm() / delta.float().norm()
        codes_changed = (merged_q[0] != q[0]).float().mean().item()
        unmerged = dequantize(quantize(merged - delta))
        drift = (unmerged - base).float().norm() / delta.float().norm()
        old_abs = F.dequantize_blockwise(q[1].absmax, q[1].state2) + q[1].offset
        new_abs = F.dequantize_blockwise(merged_q[1].absmax, merged_q[1].state2) + merged_q[1].offset
        print(json.dumps({
            "layer": name, "scale": scale,
            "delta_rms": delta.float().pow(2).mean().sqrt().item(),
            "nf4_step_rms": (base - w).float().pow(2).mean().sqrt().item(),
            "fraction_of_delta_applied": round(kept.item(), 4),
            "rel_error_vs_exact": round(err.item(), 4),
            "codes_changed": round(codes_changed, 6),
            "absmax_blocks_changed": round((old_abs != new_abs).float().mean().item(), 6),
            "unmerge_drift_rel_to_delta": round(drift.item(), 4),
        }))
