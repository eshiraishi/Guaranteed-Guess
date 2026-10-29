#!/usr/bin/env bash
# ponyo (RTX 4070 Ti SUPER, same card and stack as totoro GPU1), seed 1: generation only, scoring is done from the Mac.
# Lane: after the base HumanEval rerun, beam 8 on dev for checkpoint 106, then the sampled dev diagnostic on 106.
# The checkpoint arrives from totoro via ops/lanes/relay_seed1.sh.
cd ~/gg/ggrpo
export CUDA_MPS_PIPE_DIRECTORY=$HOME/cs8001-step0/mps_off HF_HUB_OFFLINE=1 PATH=$HOME/gg/ggrpo/.venv-train/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # without the MPS override: CUDA error 805; sampler: see sample_gpu1.sh
S=runs/eval_ponyo_seed1.status L=runs/eval_ponyo_seed1.log O=runs/full_seed1_eval
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
until grep -q "GEN_DONE base humaneval" runs/eval_ponyo.status; do sleep 60; done   # the GPU is busy until then
mkdir -p $O
c=runs/full_seed1/checkpoint-106
until [ -f $c/.relayed ]; do sleep 60; done
same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER 106 $(date -u +%FT%TZ)" >> $S; exit 1; }
echo "GEN_START 106 dev $(date -u +%FT%TZ)" >> $S
$py -m ggrpo.generate --generate.model $c --data.eval data/dev.jsonl --generate.out $O/checkpoint-106_dev.jsonl >> $L 2>&1
echo "GEN_DONE 106 dev $? $(date -u +%FT%TZ)" >> $S
echo "SAMPLE_START 106 $(date -u +%FT%TZ)" >> $S
$py runs/sample_eval.py $c data/dev.jsonl $O/106_dev_sample8.jsonl >> $L 2>&1
echo "SAMPLE_DONE 106 $? $(date -u +%FT%TZ)" >> $S
echo "LANE_DONE $(date -u +%FT%TZ)" >> $S
