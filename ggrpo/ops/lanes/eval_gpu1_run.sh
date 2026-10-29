#!/usr/bin/env bash
# totoro GPU1 (RTX 4070 Ti SUPER, modern stack), one training run (runs/NAME): eval_gpu1_arm.sh with the jobs as
# arguments, for runs that don't use its fixed 27/54/81/106 plan (the SFT baseline saves 54 and 108). Generation only,
# scoring is done from the Mac (score_run.sh). A job is STEP:SET, run in argument order on runs/NAME/checkpoint-STEP:
# SET is dev or humaneval_gpu1 / humaneval_ponyo / eval (beam 8 on data/SET.jsonl; eval is all of HumanEval-C), or
# sample (the sampled dev diagnostic, ~3 min). The GRPO arms' plan is
#   27:dev 27:sample 54:dev 54:sample 81:dev 81:sample 106:humaneval_gpu1 106:dev 106:sample
# Every GPU job holds runs/gpu1.lock and waits for an empty card, so lanes of different runs queue instead of sharing
# the 16 GB card. A job whose output is already complete is skipped, so a relaunched lane resumes.
# usage: eval_gpu1_run.sh NAME JOB...   e.g. eval_gpu1_run.sh sft_seed0 108:sample 108:dev 54:dev 54:sample
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PATH=$HOME/ggrpo/.venv/bin:$PATH \
  VLLM_USE_FLASHINFER_SAMPLER=0   # see sample_gpu1.sh
name=$1 S=runs/eval_gpu1_$1.status L=runs/eval_gpu1_$1.log O=runs/$1_eval
shift
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
gpu() {  # run "$@" holding the GPU1 lock, once no other process is on the card
  flock runs/gpu1.lock bash -c 'until [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
    exec "$@"' _ "$@"
}
has_rows() { [ "$(cat $1 2>/dev/null | wc -l)" = "$2" ]; }   # has_rows FILE ROWS: output already complete
echo "LANE_START $name $(date -u +%FT%TZ)" >> $S
mkdir -p $O
for job in "$@"; do
  n=${job%:*} set=${job#*:} c=runs/$name/checkpoint-${job%:*}
  until [ -f $c/trainer_state.json ]; do sleep 120; done; sleep 180   # let the save finish
  same_stop_tokens $c || { echo "STOP_TOKENS_DIFFER $n $(date -u +%FT%TZ)" >> $S; exit 1; }
  if [ $set = sample ]; then
    out=$O/${n}_dev_sample8.jsonl
    has_rows $out $((8 * $(wc -l < data/dev.jsonl))) && continue
    echo "SAMPLE_START $n $(date -u +%FT%TZ)" >> $S
    gpu .venv/bin/python runs/sample_eval.py $c data/dev.jsonl $out >> $L 2>&1
    echo "SAMPLE_DONE $n $? $(date -u +%FT%TZ)" >> $S
  else
    tasks=data/$set.jsonl out=$O/checkpoint-${n}_$set.jsonl
    has_rows $out $(wc -l < $tasks) && continue
    echo "GEN_START $n $set $(date -u +%FT%TZ)" >> $S
    gpu .venv/bin/python -m ggrpo.generate --generate.model $c --data.eval $tasks --generate.out $out >> $L 2>&1
    echo "GEN_DONE $n $set $? $(date -u +%FT%TZ)" >> $S
  fi
done
echo "LANE_DONE $name $(date -u +%FT%TZ)" >> $S
