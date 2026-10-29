#!/usr/bin/env bash
# ponyo (RTX 4070 Ti SUPER, same card and stack as totoro GPU1), one training run (runs/NAME): eval_ponyo_arm.sh with
# the jobs as arguments. Generation only, scoring is done from the Mac (score_run.sh). A job is STEP:SET, beam 8 on
# data/SET.jsonl for runs/NAME/checkpoint-STEP, in argument order: humaneval_ponyo (the larger half of the split with
# GPU1, as for the GRPO arms: 106:humaneval_ponyo) or eval (all of HumanEval-C, for the SFT baseline: 108:eval).
# Checkpoints arrive from totoro via ops/lanes/relay_run.sh. GPU jobs hold runs/gpu.lock and wait for an empty card.
# usage: eval_ponyo_run.sh NAME JOB...   e.g. eval_ponyo_run.sh sft_seed0 108:eval
cd ~/gg/ggrpo
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off HF_HUB_OFFLINE=1 PATH=$HOME/gg/ggrpo/.venv-train/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # without the MPS override: CUDA error 805; sampler: see sample_gpu1.sh
name=$1 S=runs/eval_ponyo_$1.status L=runs/eval_ponyo_$1.log O=runs/$1_eval
shift
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
for job in "$@"; do
  n=${job%:*} set=${job#*:} c=runs/$name/checkpoint-${job%:*}
  tasks=data/$set.jsonl out=$O/checkpoint-${n}_$set.jsonl
  until [ -f $c/.relayed ]; do sleep 60; done
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u +%FT%TZ)" >> $S; exit 1; }
  [ "$(cat $out 2>/dev/null | wc -l)" = "$(wc -l < $tasks)" ] && continue
  echo "GEN_START $n $set $(date -u +%FT%TZ)" >> $S
  gpu $py -m ggrpo.generate --generate.model $c --data.eval $tasks --generate.out $out >> $L 2>&1
  echo "GEN_DONE $n $set $? $(date -u +%FT%TZ)" >> $S
done
echo "LANE_DONE $name $(date -u +%FT%TZ)" >> $S
