#!/usr/bin/env bash
# Runs on the Mac (nohup), seed 1: relay.sh + relay_cluster.sh for runs/full_seed1/checkpoint-106 (see those for why).
# 1. ponyo pulls the eval files from totoro (for the ponyo lane). 2. ponyo pushes the weights to student-lab, which
# builds a transformers-4.50 view dir, checks it, and queues BringUp beam8 + greedy with seed 0's shards.
keep="--include=*.json --include=*.jinja --include=model.safetensors --exclude=*"
src=ggrpo/runs/full_seed1/checkpoint-106 ck=gg/ggrpo/runs/full_seed1/checkpoint-106
until ssh -o BatchMode=yes -o ConnectTimeout=20 totoro.mbzu.ae "test -f $src/trainer_state.json"; do sleep 120; done
sleep 180   # let the save finish
until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo "mkdir -p $ck && rsync -a $keep totoro.mbzu.ae:$src/ $ck/ && touch $ck/.relayed"; do
  echo "relay failed $(date -u +%FT%TZ), retrying"; sleep 120
done
echo "RELAYED 106 $(date -u +%FT%TZ)"
ssh -o BatchMode=yes student-lab 'mkdir -p ~/cs8001-bringup/trained_seed1/checkpoint-106'
until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo \
    "rsync -a $ck/model.safetensors enzo.shiraishi@login-student-lab.mbzu.ae:cs8001-bringup/trained_seed1/checkpoint-106/"; do
  echo "push failed $(date -u +%FT%TZ), retrying"; sleep 120
done
echo "PUSHED $(date -u +%FT%TZ)"
ssh -o BatchMode=yes student-lab 'set -e; cd ~/cs8001-bringup
base=$HOME/.cache/huggingface/hub/models--ahmedheakl--gg-armv8-O2/snapshots/0444a378f384300187c7f43914fc986a81ff051a
view=trained_seed1/view-106; mkdir -p $view
cp -L $base/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json} $view/
ln -sfn $PWD/trained_seed1/checkpoint-106/model.safetensors $view/model.safetensors
.venv/bin/python -c "
from transformers import AutoConfig; from safetensors import safe_open
c = AutoConfig.from_pretrained(\"$view\"); assert c.rope_theta == 1e6, c.rope_theta
n = len(safe_open(\"$view/model.safetensors\", \"pt\").keys()); assert n == 338, n
print(\"VIEW_OK rope_theta\", c.rope_theta, \"tensors\", n)"
for s in a b; do sbatch -t 6:00:00 -J s1b8-$s job.sh trained_seed1/beam8_$s.jsonl beam8 "$(cat trained_beam8_$s.txt)" --model $view; done
for s in a b; do sbatch -t 3:00:00 -J s1g-$s job.sh trained_seed1/greedy_$s.jsonl released "$(cat greedy_$s.txt)" --model $view; done
squeue -u enzo.shiraishi' && echo "SUBMITTED $(date -u +%FT%TZ)" || echo "SUBMIT_FAILED $(date -u +%FT%TZ)"
