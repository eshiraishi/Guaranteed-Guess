#!/usr/bin/env bash
# Runs on the Mac (nohup), one training run (runs/NAME): relay_arm.sh with the final step as an argument (106 for
# the GRPO arms, 108 for SFT) (see relay.sh, relay_cluster.sh).
# 1. ponyo pulls checkpoint-FINAL's eval files from totoro (for the ponyo lane). 2. ponyo pushes the weights to
# student-lab, which builds a transformers-4.50 view dir, checks it, and queues BringUp beam8 (first: it is the paper's
# metric and the longer job) and then greedy, with seed 0's shards; QOS runs at most 2 jobs at a time.
# usage: relay_run.sh NAME FINAL   e.g. relay_run.sh sft_seed0 108
name=$1 n=$2
keep="--include=*.json --include=*.jinja --include=model.safetensors --exclude=*"
src=ggrpo/runs/$name/checkpoint-$n ck=gg/ggrpo/runs/$name/checkpoint-$n
until ssh -o BatchMode=yes -o ConnectTimeout=20 totoro.mbzu.ae "test -f $src/trainer_state.json"; do sleep 120; done
sleep 180   # let the save finish
until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo "mkdir -p $ck && rsync -a $keep totoro.mbzu.ae:$src/ $ck/ && touch $ck/.relayed"; do
  echo "relay failed $(date -u +%FT%TZ), retrying"; sleep 120
done
echo "RELAYED $name $n $(date -u +%FT%TZ)"
until ssh -o BatchMode=yes -o ConnectTimeout=20 student-lab "mkdir -p ~/cs8001-bringup/$name/checkpoint-$n"; do sleep 120; done
until ssh -A -o BatchMode=yes -o ConnectTimeout=20 ponyo \
    "rsync -a $ck/model.safetensors enzo.shiraishi@login-student-lab.mbzu.ae:cs8001-bringup/$name/checkpoint-$n/"; do
  echo "push failed $(date -u +%FT%TZ), retrying"; sleep 120
done
echo "PUSHED $name $(date -u +%FT%TZ)"
ssh -o BatchMode=yes student-lab "set -e; cd ~/cs8001-bringup; name=$name n=$n"'
base=$HOME/.cache/huggingface/hub/models--ahmedheakl--gg-armv8-O2/snapshots/0444a378f384300187c7f43914fc986a81ff051a
view=$name/view-$n; mkdir -p $view
cp -L $base/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json} $view/
ln -sfn $PWD/$name/checkpoint-$n/model.safetensors $view/model.safetensors
.venv/bin/python -c "
from transformers import AutoConfig; from safetensors import safe_open
c = AutoConfig.from_pretrained(\"$view\"); assert c.rope_theta == 1e6, c.rope_theta
n = len(safe_open(\"$view/model.safetensors\", \"pt\").keys()); assert n == 338, n
print(\"VIEW_OK rope_theta\", c.rope_theta, \"tensors\", n)"
for s in a b; do sbatch -t 6:00:00 -J $name-b8-$s job.sh $name/beam8_$s.jsonl beam8 "$(cat trained_beam8_$s.txt)" --model $view; done
for s in a b; do sbatch -t 3:00:00 -J $name-g-$s job.sh $name/greedy_$s.jsonl released "$(cat greedy_$s.txt)" --model $view; done
squeue -u enzo.shiraishi' && echo "SUBMITTED $name $(date -u +%FT%TZ)" || echo "SUBMIT_FAILED $name $(date -u +%FT%TZ)"
