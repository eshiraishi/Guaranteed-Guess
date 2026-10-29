#!/usr/bin/env bash
# Runs on the Mac (nohup): when ponyo has checkpoint-106 (relay.sh), push its weights ponyo -> student-lab, build the
# transformers-4.50 view dir there, check it, and queue the trained-model BringUp runs (beam8 + greedy, 2 shards each;
# QOS runs at most 2 at a time).
# Why a view dir: TRL/transformers 5.17 checkpoints store rope_parameters, which 4.50 ignores (silently rope_theta 1e4
# instead of 1e6). Base config/generation/tokenizer files + the trained model.safetensors gives logits bit-identical
# to 5.17 in fp32 (verified on runs/probe_full, Oct 22).
ck=gg/ggrpo/runs/full/checkpoint-106
until ssh -o BatchMode=yes -o ConnectTimeout=20 ponyo "test -f $ck/.relayed"; do sleep 120; done
ssh -o BatchMode=yes student-lab 'mkdir -p ~/cs8001-bringup/trained/checkpoint-106'
until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo \
    "rsync -a $ck/model.safetensors enzo.shiraishi@login-student-lab.mbzu.ae:cs8001-bringup/trained/checkpoint-106/"; do
  echo "push failed $(date -u +%FT%TZ), retrying"; sleep 120
done
echo "PUSHED $(date -u +%FT%TZ)"
ssh -o BatchMode=yes student-lab 'set -e; cd ~/cs8001-bringup
base=$HOME/.cache/huggingface/hub/models--ahmedheakl--gg-armv8-O2/snapshots/0444a378f384300187c7f43914fc986a81ff051a
view=trained/view-106; mkdir -p $view
cp -L $base/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json} $view/
ln -sfn $PWD/trained/checkpoint-106/model.safetensors $view/model.safetensors
.venv/bin/python -c "
from transformers import AutoConfig; from safetensors import safe_open
c = AutoConfig.from_pretrained(\"$view\"); assert c.rope_theta == 1e6, c.rope_theta
n = len(safe_open(\"$view/model.safetensors\", \"pt\").keys()); assert n == 338, n
print(\"VIEW_OK rope_theta\", c.rope_theta, \"tensors\", n)"
# shards: beam8 balanced on the base run'"'"'s RTX 5000 Ada times; greedy reuses the base greedy shards
.venv/bin/python - <<PY
import glob, json
rows = [json.loads(l) for f in glob.glob("base/beam8_*.jsonl") + glob.glob("test/*.jsonl") for l in open(f)]
assert len({r["file"] for r in rows}) == 66, len(rows)
a, b, sa, sb = [], [], 0, 0
for r in sorted(rows, key=lambda r: -r["seconds"]):
    if sa <= sb: a.append(r["file"]); sa += r["seconds"]
    else: b.append(r["file"]); sb += r["seconds"]
open("trained_beam8_a.txt", "w").write(",".join(a)); open("trained_beam8_b.txt", "w").write(",".join(b))
print("beam8 shards", len(a), len(b), round(sa / 3600, 2), round(sb / 3600, 2))
PY
for s in a b; do sbatch -t 6:00:00 -J tb8-$s job.sh trained/beam8_$s.jsonl beam8 "$(cat trained_beam8_$s.txt)" --model $view; done
for s in a b; do sbatch -t 3:00:00 -J tg-$s job.sh trained/greedy_$s.jsonl released "$(cat greedy_$s.txt)" --model $view; done
squeue -u enzo.shiraishi' && echo "SUBMITTED $(date -u +%FT%TZ)" || echo "SUBMIT_FAILED $(date -u +%FT%TZ)"
