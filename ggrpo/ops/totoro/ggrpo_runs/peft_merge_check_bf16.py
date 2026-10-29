"""Same as peft_merge_check.py for the LoRA arm: merge into the bf16 base (TRL's vLLM sync), measure what reaches vLLM."""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

REPO, REV = "ahmedheakl/gg-armv8-O2", "0444a378f384300187c7f43914fc986a81ff051a"
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, sys.argv[1])
layers = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}
for scale in [1, 10]:
    for m in layers.values():
        m.scaling["default"] = 32 * scale
    original = {n: m.base_layer.weight.data.clone() for n, m in layers.items()}
    delta = {n: m.get_delta_weight("default").float() for n, m in layers.items()}
    dnorm = sum(d.norm() ** 2 for d in delta.values()).sqrt()
    model.merge_adapter()
    reach = sum(((m.base_layer.weight.data - original[n]).float() * delta[n]).sum() for n, m in layers.items()) / dnorm**2
    changed = sum((m.base_layer.weight.data != original[n]).sum().item() for n, m in layers.items()) / sum(o.numel() for o in original.values())
    model.unmerge_adapter()
    out = {"scale": scale, "delta_rms": (dnorm**2 / sum(d.numel() for d in delta.values())).sqrt().item(),
           "weights_changed_by_merge": changed, "fraction_of_delta_reaching_vllm": reach.item()}
    for cycle in range(2, 11):
        model.merge_adapter()
        model.unmerge_adapter()
    out["base_drift_rel_to_delta_after_10_cycles"] = (sum((m.base_layer.weight.data - original[n]).float().norm() ** 2 for n, m in layers.items()).sqrt() / dnorm).item()
    print(out)
    for n, m in layers.items():
        m.base_layer.weight.data.copy_(original[n])
