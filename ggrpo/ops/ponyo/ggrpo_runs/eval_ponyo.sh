#!/usr/bin/env bash
# ponyo (RTX 4070 Ti SUPER, same card and stack as totoro GPU1): generation only, scoring is done from the Mac.
# Lane: after seed 1 of the authors' script, a cross-machine identity check (base model, first 8 dev tasks, compared with
# totoro GPU1's base dev rows), then dev on checkpoints 81 and 106. Checkpoints arrive from totoro via ops/lanes/relay.sh.
cd ~/gg/ggrpo
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off HF_HUB_OFFLINE=1   # without the MPS override: CUDA error 805
S=runs/eval_ponyo.status L=runs/eval_ponyo.log
py=.venv-train/bin/python
same_stop_tokens() { $py - $1 >> $L 2>&1 <<PY
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
until grep -q SEED1_DONE runs/paper_seed1.log; do sleep 60; done   # the GPU is busy with seed 1 until then
mkdir -p runs/xcheck runs/full_eval
head -n 8 data/dev.jsonl > runs/xcheck/dev8.jsonl
$py -m ggrpo.generate --data.eval runs/xcheck/dev8.jsonl --generate.out runs/xcheck/dev8_base_ponyo.jsonl >> $L 2>&1
echo "XCHECK_DONE $? $(date -u +%FT%TZ)" >> $S
for n in 81 106; do
  c=runs/full/checkpoint-$n
  until [ -f $c/.relayed ]; do sleep 60; done
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "GEN_START $n dev $(date -u +%FT%TZ)" >> $S
  $py -m ggrpo.generate --generate.model $c --data.eval data/dev.jsonl --generate.out runs/full_eval/checkpoint-${n}_dev.jsonl >> $L 2>&1
  echo "GEN_DONE $n dev $? $(date -u +%FT%TZ)" >> $S
done
echo "LANE_DONE $(date -u +%FT%TZ)" >> $S
