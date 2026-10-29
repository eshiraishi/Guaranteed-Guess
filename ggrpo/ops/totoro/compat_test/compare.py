"""compare.py REF_PREFIX OTHER_PREFIX [n_prompt]: logits agreement over all positions (and prompt ids across stacks)."""
import json
import sys

import torch

r, o = sys.argv[1], sys.argv[2]
a, b = torch.load(r + ".logits.pt"), torch.load(o + ".logits.pt")
ja, jb = json.load(open(r + ".json")), json.load(open(o + ".json"))
d = (a - b).abs()
la, lb = a.log_softmax(-1), b.log_softmax(-1)
agree = a.argmax(-1) == b.argmax(-1)
pos_max = d.max(-1).values
first_bad = int((~agree).nonzero()[0]) if (~agree).any() else None
print(json.dumps({
    "ref": f"{ja['transformers']} {ja['model']} {ja['dtype']}", "other": f"{jb['transformers']} {jb['model']} {jb['dtype']}",
    "positions": a.shape[0], "max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
    "max_abs_logprob_diff": (la - lb).abs().max().item(),
    "argmax_agree": f"{int(agree.sum())}/{agree.numel()}", "first_disagree_pos": first_bad,
    "max_abs_diff_pos0": pos_max[0].item(), "max_abs_diff_pos1_10": pos_max[1:11].max().item(),
    "ref_logit_absmax": a.abs().max().item(),
    "prompt_ids_equal_across_stacks": [p["ids"] == q["ids"] for p, q in zip(ja["prompts"], jb["prompts"])],
}, indent=1))
