"""Load a model dir with whatever transformers this venv has (CPU only) and report config, tokenizer, weights, rope,
resolved generation settings; save next-token logits for the long input. Usage:
probe.py MODEL_DIR --base BASE_DIR --inputs inputs.json --out PREFIX [--dtype fp32|bf16] [--threads N]"""
import argparse
import json
import os
import time
import tomllib

import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--base", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--toml", default=os.path.expanduser("~/ggrpo/config.toml"))
a = ap.parse_args()
torch.set_num_threads(a.threads)
v5 = int(transformers.__version__.split(".")[0]) >= 5
rep = {"transformers": transformers.__version__, "torch": torch.__version__, "model": a.model, "dtype": a.dtype}
inp = json.load(open(a.inputs))

# --- config
cfg = AutoConfig.from_pretrained(a.model)
rep["config"] = {k: getattr(cfg, k, "<absent>") for k in [
    "rope_theta", "rope_parameters", "rope_scaling", "sliding_window", "use_sliding_window", "max_window_layers",
    "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id", "use_cache"]}

# --- generation_config.json, model vs base
gm, gb = GenerationConfig.from_pretrained(a.model).to_dict(), GenerationConfig.from_pretrained(a.base).to_dict()
rep["genconfig_diff_vs_base"] = {k: (gm.get(k), gb.get(k)) for k in sorted(set(gm) | set(gb))
                                 if gm.get(k) != gb.get(k) and k != "transformers_version"}

# --- tokenizer and chat template, model dir vs base dir (eval path: template as text, then tok([text]))
tm, tb = AutoTokenizer.from_pretrained(a.model), AutoTokenizer.from_pretrained(a.base)


def attrs(t):
    return {"class": type(t).__name__, "len": len(t), "vocab_size": t.vocab_size, "bos": t.bos_token,
            "eos": [t.eos_token, t.eos_token_id], "pad": [t.pad_token, t.pad_token_id], "padding_side": t.padding_side,
            "truncation_side": t.truncation_side, "special_ids": sorted(t.all_special_ids),
            "model_max_length": t.model_max_length, "has_chat_template": t.chat_template is not None}


rep["tok_model"], rep["tok_base"] = attrs(tm), attrs(tb)
rep["chat_template_equal"] = tm.chat_template == tb.chat_template
rep["prompts"] = []
for p in inp["prompts"]:
    sm = tm.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
    sb = tb.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
    im, ib = tm([sm]).input_ids[0], tb([sb]).input_ids[0]
    rep["prompts"].append({"task_id": p["task_id"], "text_equal": sm == sb, "ids_equal": im == ib, "n": len(im),
                           "ids": im, "head": sm[:80], "tail": sm[-40:]})
lm = tm([tm.apply_chat_template(inp["long"]["messages"], tokenize=False)]).input_ids[0]
rep["long_ids_equal_to_saved"] = lm == inp["long"]["ids"]
completion = tm("```armasm\n\tmov\tw0, #1\n\tret\n```").input_ids + [151645, 151643]
rep["decode_skip_special"] = {"model": tm.decode(completion, skip_special_tokens=True),
                              "base": tb.decode(completion, skip_special_tokens=True)}
rep["decode_skip_special_equal"] = rep["decode_skip_special"]["model"] == rep["decode_skip_special"]["base"]

# --- model weights
dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
t0 = time.time()
model, info = AutoModelForCausalLM.from_pretrained(
    a.model, attn_implementation="sdpa", output_loading_info=True, **({"dtype": dt} if v5 else {"torch_dtype": dt}))
model.eval()
rep["load_seconds"] = round(time.time() - t0, 1)
rep["loading_info"] = {k: sorted(str(x) for x in v) for k, v in info.items()}
rep["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
rep["lm_head_tied"] = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
sd = model.state_dict()
with safe_open(os.path.join(a.model, "model.safetensors"), "pt") as f:
    fkeys = set(f.keys())
    maxdiff = max((sd[k].float() - f.get_tensor(k).to(dt).float()).abs().max().item() for k in fkeys if k in sd)
rep["weights_vs_file"] = {"file_tensors": len(fkeys), "file_not_in_model": sorted(fkeys - set(sd)),
                          "model_not_in_file": sorted(set(sd) - fkeys), "max_abs_diff_after_cast": maxdiff}
del sd

# --- effective rope: inv_freq[i] = theta ** (-2i/d)  =>  theta = inv_freq[1] ** (-d/2)
rot = model.model.rotary_emb
inv = rot.inv_freq.double()
d = 2 * inv.numel()
rep["rope"] = {"rope_type": getattr(rot, "rope_type", None), "attention_scaling": getattr(rot, "attention_scaling", None),
               "head_dim": d, "theta_from_inv_freq": float(inv[1] ** (-d / 2))}

# --- generation settings each eval would actually use (no generation, just the merge)
modes = tomllib.load(open(a.toml, "rb"))["generate"]
fields = ["do_sample", "num_beams", "temperature", "top_k", "top_p", "repetition_penalty", "eos_token_id",
          "pad_token_id", "bos_token_id"]
resolved = {}
for m in ["beam_search", "sample"]:   # ggrpo.generate: explicit GenerationConfig object
    g, _ = model._prepare_generation_config(GenerationConfig.from_dict(modes[m] | {"max_new_tokens": 100}))
    resolved[f"ggrpo.{m}"] = {k: getattr(g, k) for k in fields}
for m, kw in {"released": dict(max_new_tokens=100, do_sample=False),
              "beam8": dict(max_new_tokens=100, num_beams=8, num_return_sequences=1, do_sample=False)}.items():
    g, _ = model._prepare_generation_config(None, **kw)   # gen_bringup.py: kwargs only
    resolved[f"bringup.{m}"] = {k: getattr(g, k) for k in fields}
rep["resolved_generation"] = resolved

# --- logits on the long input
ids = torch.tensor([inp["long"]["ids"]])
t0 = time.time()
with torch.inference_mode():
    logits = model(input_ids=ids, use_cache=False).logits[0].float()
rep["forward_seconds"] = round(time.time() - t0, 1)
rep["n_positions"] = logits.shape[0]
torch.save(logits, a.out + ".logits.pt")
json.dump(rep, open(a.out + ".json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rep.items() if k != "prompts"}, indent=1, default=str))
print("prompts", [(p["task_id"], p["n"], p["text_equal"], p["ids_equal"]) for p in rep["prompts"]])
