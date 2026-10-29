#!/usr/bin/env bash
# totoro GPU1 (RTX 4070 Ti SUPER, modern stack): generation only, scoring is done from the Mac against the VM.
# Base references on this card: HumanEval runs/modern/beam_search.jsonl (72/164), dev runs/modern/dev_beam_search.jsonl.
# Lane: dev on checkpoints 27 and 54, then HumanEval-C on 106 (the headline). ponyo (same card) takes dev 81 and 106.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
S=runs/eval_gpu1.status L=runs/eval_gpu1.log
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
echo "LANE_START $(date -u +%FT%TZ)" >> $S
until grep -q DEV_BASE_DONE runs/modern_dev_beam_search.log; do sleep 60; done   # GPU1 is busy with the base dev run until then
mkdir -p runs/full_eval
for job in 27:dev 54:dev 106:humaneval; do
  n=${job%:*} set=${job#*:} c=runs/full/checkpoint-${job%:*}
  until [ -f $c/trainer_state.json ]; do sleep 120; done; sleep 180   # let the save finish
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u +%FT%TZ)" >> $S; exit 1; }
  tasks=data/eval.jsonl; [ $set = dev ] && tasks=data/dev.jsonl
  echo "GEN_START $n $set $(date -u +%FT%TZ)" >> $S
  .venv/bin/python -m ggrpo.generate --generate.model $c --data.eval $tasks --generate.out runs/full_eval/checkpoint-${n}_$set.jsonl >> $L 2>&1
  echo "GEN_DONE $n $set $? $(date -u +%FT%TZ)" >> $S
done
echo "LANE_DONE $(date -u +%FT%TZ)" >> $S
