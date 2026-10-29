"""Pick 3 short chat prompts and one ~1.5-2k-token full conversation from eval.jsonl; token ids from the BASE tokenizer."""
import json
import sys

from transformers import AutoTokenizer

base, eval_jsonl, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(base)
rows = [json.loads(l) for l in open(eval_jsonl)]


def prompt_ids(msgs, gen):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    return tok([s]).input_ids[0]


lens = sorted((len(prompt_ids(r["prompt"], True)), i) for i, r in enumerate(rows))
picks = [lens[0][1], lens[len(lens) // 4][1], lens[len(lens) // 2][1]]
convs = []
for i, r in enumerate(rows):
    msgs = r["prompt"] + [{"role": "assistant", "content": "```armasm\n" + r["arm_ref"] + "```"}]
    convs.append((abs(len(prompt_ids(msgs, False)) - 1800), i, msgs))
_, li, lmsgs = min(convs, key=lambda c: c[:2])
long_ids = prompt_ids(lmsgs, False)
json.dump({
    "prompts": [{"task_id": rows[i]["task_id"], "messages": rows[i]["prompt"]} for i in picks],
    "long": {"task_id": rows[li]["task_id"], "messages": lmsgs, "ids": long_ids,
             "n_prompt": len(prompt_ids(rows[li]["prompt"], True))},
}, open(out, "w"))
print("short", [(rows[i]["task_id"], n) for n, i in lens if i in picks], "long", rows[li]["task_id"], len(long_ids))
