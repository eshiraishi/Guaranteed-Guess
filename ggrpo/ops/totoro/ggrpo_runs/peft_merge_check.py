"""Run PEFT's own merge_adapter/unmerge_adapter on the QLoRA model (what TRL's vLLM sync calls every step).

Reports, over all LoRA-targeted Linear4bit layers:
- codes_changed_by_merge: fraction of packed 4-bit bytes that differ after merge (what TRL pushes to vLLM)
- update_reaching_vllm: |deq(merged codes, vLLM's original absmax) - deq(base)| / |delta|   (vLLM keeps its own absmax)
- base_drift_after_cycles: |deq(base after N merge+unmerge cycles) - deq(original base)| / |delta|
"""

import sys

import bitsandbytes.functional as F
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REPO, REV = "ahmedheakl/gg-armv8-O2", "0444a378f384300187c7f43914fc986a81ff051a"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=bnb, dtype=torch.bfloat16,
                                             device_map={"": 0})
model = PeftModel.from_pretrained(model, sys.argv[1])
layers = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}


def snapshot():
    return {n: (m.base_layer.weight.data.clone(), F.dequantize_4bit(m.base_layer.weight.data, m.base_layer.weight.quant_state),
             m.base_layer.weight.quant_state)
            for n, m in layers.items()}


def deltas():
    return {n: m.get_delta_weight("default").float() for n, m in layers.items()}


original, delta = snapshot(), deltas()
dnorm = sum(d.norm() ** 2 for d in delta.values()).sqrt()
model.merge_adapter()
merged = snapshot()
changed = sum((merged[n][0] != original[n][0]).sum().item() for n in layers) / sum(original[n][0].numel() for n in layers)
# vLLM decodes the pushed codes with the absmax it computed at its own load (= the original quant_state)
reach = sum(((F.dequantize_4bit(merged[n][0], original[n][2]) - original[n][1]).float() * delta[n]).sum() for n in layers) / dnorm**2
applied = sum(((merged[n][1] - original[n][1]).float() * delta[n]).sum() for n in layers) / dnorm**2
print({"delta_rms": (dnorm**2 / sum(d.numel() for d in delta.values())).sqrt().item(),
       "codes_changed_by_merge": changed, "fraction_of_delta_in_merged_weights": applied.item(),
       "fraction_of_delta_reaching_vllm": reach.item()})
model.unmerge_adapter()
for cycle in range(1, 11):
    if cycle > 1:
        model.merge_adapter()
        model.unmerge_adapter()
    now = snapshot()
    drift = sum((now[n][1] - original[n][1]).float().norm() ** 2 for n in layers).sqrt() / dnorm
    codes = sum((now[n][0] != original[n][0]).sum().item() for n in layers) / sum(original[n][0].numel() for n in layers)
    if cycle in (1, 2, 5, 10):
        print({"cycles": cycle, "base_drift_rel_to_delta": round(drift.item(), 4), "base_codes_changed": codes})
