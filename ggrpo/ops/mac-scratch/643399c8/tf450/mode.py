"""What does eval/eval_huamaneval.py's generate() call resolve to under transformers 4.50.0?
Tiny random Qwen2 with the real checkpoint's generation_config; same kwargs as the authors' script."""
import torch, transformers
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
snap = __import__("sys").argv[1]
cfg = AutoConfig.from_pretrained(snap); cfg.num_hidden_layers = 1; cfg.hidden_size = 64; cfg.intermediate_size = 128; cfg.num_attention_heads = 2; cfg.num_key_value_heads = 1
m = AutoModelForCausalLM.from_config(cfg)
m.generation_config = GenerationConfig.from_pretrained(snap)
gc, _ = m._prepare_generation_config(None, max_new_tokens=100, num_beams=8, num_return_sequences=1)
print("transformers", transformers.__version__, "| mode:", gc.get_generation_mode(),
      {k: getattr(gc, k) for k in ["do_sample", "num_beams", "temperature", "top_k", "top_p", "repetition_penalty"]})
ids = torch.tensor([[151644, 872, 198, 100, 200, 300]])
outs = []
for seed in (0, 1):
    torch.manual_seed(seed)
    outs.append(m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, num_beams=8, num_return_sequences=1)[0].tolist())
print("seed 0 vs seed 1 identical:", outs[0] == outs[1])
torch.manual_seed(0); a = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, num_beams=8, do_sample=False)[0].tolist()
torch.manual_seed(1); b = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, num_beams=8, do_sample=False)[0].tolist()
print("with do_sample=False, seed 0 vs 1 identical:", a == b)
