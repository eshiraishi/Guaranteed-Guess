#!/usr/bin/env bash
# ponyo (RTX 4070 Ti SUPER, same card and stack as totoro GPU1), one training run (runs/NAME): generation only, scoring
# is done from the Mac. Lane: HumanEval-C beam 8 on checkpoint 106 for data/humaneval_ponyo.jsonl, the larger half of
# the split with totoro GPU1 (see eval_gpu1_arm.sh). The checkpoint arrives from totoro via ops/lanes/relay_arm.sh.
# GPU jobs hold runs/gpu.lock and wait for an empty card, as on GPU1.
# usage: eval_ponyo_arm.sh NAME
cd ~/gg/ggrpo
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off HF_HUB_OFFLINE=1 PATH=$HOME/gg/ggrpo/.venv-train/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # without the MPS override: CUDA error 805; sampler: see sample_gpu1.sh
name=$1 S=runs/eval_ponyo_$1.status L=runs/eval_ponyo_$1.log O=runs/$1_eval
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
gpu() {  # run "$@" holding the GPU lock, once no other process is on the card
  flock runs/gpu.lock bash -c 'until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
    exec "$@"' _ "$@"
}
echo "LANE_START $name $(date -u +%FT%TZ)" >> $S
mkdir -p $O
c=runs/$name/checkpoint-106 tasks=data/humaneval_ponyo.jsonl out=$O/checkpoint-106_humaneval_ponyo.jsonl
until [ -f $c/.relayed ]; do sleep 60; done
same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER 106 $(date -u +%FT%TZ)" >> $S; exit 1; }
if [ "$(cat $out 2>/dev/null | wc -l)" != "$(wc -l < $tasks)" ]; then
  echo "GEN_START 106 humaneval_ponyo $(date -u +%FT%TZ)" >> $S
  gpu $py -m ggrpo.generate --generate.model $c --data.eval $tasks --generate.out $out >> $L 2>&1
  echo "GEN_DONE 106 humaneval_ponyo $? $(date -u +%FT%TZ)" >> $S
fi
echo "LANE_DONE $name $(date -u +%FT%TZ)" >> $S
