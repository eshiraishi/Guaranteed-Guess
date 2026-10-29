#!/usr/bin/env bash
# totoro GPU1 (RTX 4070 Ti SUPER, modern stack), seed 1 (runs/full_seed1): generation only, scoring is done from the Mac.
# Same evals as seed 0: beam 8 on dev for checkpoints 27/54/81 and on HumanEval-C for 106; ponyo takes dev 106.
# After each beam job, the sampled dev diagnostic (ops/lanes/sample_eval.py, ~3 min) on the same checkpoint, while the
# GPU would otherwise wait for the next save.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # see sample_gpu1.sh
S=runs/eval_gpu1_seed1.status L=runs/eval_gpu1_seed1.log O=runs/full_seed1_eval
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
mkdir -p $O
for job in 27:dev 54:dev 81:dev 106:humaneval; do
  n=${job%:*} set=${job#*:} c=runs/full_seed1/checkpoint-${job%:*}
  until [ -f $c/trainer_state.json ]; do sleep 120; done; sleep 180   # let the save finish
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u +%FT%TZ)" >> $S; exit 1; }
  tasks=data/eval.jsonl; [ $set = dev ] && tasks=data/dev.jsonl
  echo "GEN_START $n $set $(date -u +%FT%TZ)" >> $S
  .venv/bin/python -m ggrpo.generate --generate.model $c --data.eval $tasks --generate.out $O/checkpoint-${n}_$set.jsonl >> $L 2>&1
  echo "GEN_DONE $n $set $? $(date -u +%FT%TZ)" >> $S
  [ $n = 106 ] && break   # ponyo samples 106
  echo "SAMPLE_START $n $(date -u +%FT%TZ)" >> $S
  .venv/bin/python runs/sample_eval.py $c data/dev.jsonl $O/${n}_dev_sample8.jsonl >> $L 2>&1
  echo "SAMPLE_DONE $n $? $(date -u +%FT%TZ)" >> $S
done
echo "LANE_DONE $(date -u +%FT%TZ)" >> $S
