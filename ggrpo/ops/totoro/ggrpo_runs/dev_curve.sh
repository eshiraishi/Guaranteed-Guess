#!/usr/bin/env bash
# Dev curve on GPU1 with the base HumanEval run's settings (beam 8, cap 32000 - prompt, modern stack):
# score base dev, then for checkpoints 27/54/81/106 of runs/full generate + score dev; HumanEval on 106 before its dev.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
S=runs/dev_curve.status L=runs/dev_curve.log
score() { .venv/bin/python -m ggrpo.score --score.tasks $1 --score.predictions $2 --score.out ${2%.jsonl}.scored.jsonl >> $L 2>&1; }
same_stop_tokens() { .venv/bin/python - $1 >> $L 2>&1 <<PY
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
base = json.loads(Path(hf_hub_download("ahmedheakl/gg-armv8-O2", "generation_config.json", revision="0444a378f384300187c7f43914fc986a81ff051a")).read_text())
ckpt = json.loads((Path(sys.argv[1]) / "generation_config.json").read_text())
diff = {k: (base.get(k), ckpt.get(k)) for k in ("eos_token_id", "pad_token_id") if base.get(k) != ckpt.get(k)}
print("stop tokens", sys.argv[1], "diff", diff)
sys.exit(bool(diff))
PY
}
echo "WATCH_START $(date -u)" >> $S
until grep -q DEV_BASE_DONE runs/modern_dev_beam_search.log; do sleep 60; done
score data/dev.jsonl runs/modern/dev_beam_search.jsonl; echo "BASE_DEV_SCORED $? $(date -u)" >> $S
mkdir -p runs/full_eval
for n in 27 54 81 106; do
  c=runs/full/checkpoint-$n
  until [ -f $c/trainer_state.json ]; do sleep 120; done; sleep 180   # let the save finish
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u)" >> $S; exit 1; }
  if [ $n = 106 ]; then
    out=runs/full_eval/checkpoint-106_humaneval.jsonl
    .venv/bin/python -m ggrpo.generate --generate.model $c --generate.out $out >> $L 2>&1; echo "HUMANEVAL_GEN $n $? $(date -u)" >> $S
    score data/eval.jsonl $out; echo "HUMANEVAL_SCORED $n $? $(date -u)" >> $S
  fi
  out=runs/full_eval/checkpoint-${n}_dev.jsonl
  .venv/bin/python -m ggrpo.generate --generate.model $c --data.eval data/dev.jsonl --generate.out $out >> $L 2>&1; echo "DEV_GEN $n $? $(date -u)" >> $S
  score data/dev.jsonl $out; echo "DEV_SCORED $n $? $(date -u)" >> $S
done
echo "WATCH_DONE $(date -u)" >> $S
