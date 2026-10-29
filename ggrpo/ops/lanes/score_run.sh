#!/usr/bin/env bash
# Runs on the Mac (nohup), one training run (runs/NAME): score_arm.sh with the jobs as arguments, for the lanes of
# eval_gpu1_run.sh / eval_ponyo_run.sh. Scores each eval file as soon as its lane reports it done, into the paths
# registered under [track.runs.*] in config.toml (OUT), through the VM's server. A job is STEP:SET: dev (beam 8, GPU1),
# sample (sampled dev, GPU1), eval (all of HumanEval-C, ponyo) or split (GPU1's and ponyo's halves concatenated); the
# HumanEval file is checked against data/eval.jsonl before scoring. Copies checkpoint-FINAL's trainer_state.json once
# that checkpoint exists. Already scored files are skipped, so a relaunch resumes. BringUp is scored by hand.
# usage: score_run.sh NAME OUT FINAL JOB...
#   e.g. score_run.sh sft_seed0 results/grpo_eval/sft-seed0 108 108:sample 108:eval 108:dev 54:dev 54:sample
cd "$(dirname "$0")/../.."
name=$1 out=$2 final=$3 url=http://192.168.64.2:8765/score
src=ggrpo/runs/${name}_eval
mkdir -p $out results/training/${out##*/}
shift 3
ok() { grep -q "$2" <(ssh -o BatchMode=yes -o ConnectTimeout=20 $1 "cat $3" 2>/dev/null); }   # ok HOST PATTERN FILE
beam() {  # beam TASKS PREDICTIONS: ggrpo.score into the .scored.jsonl next to it, after a row and id check
  uv run --frozen python -c "
import polars as pl, sys
p, t = pl.read_ndjson('$2'), pl.read_ndjson('$1')
assert p.height == t.height and set(p['task_id']) == set(t['task_id']), (p.height, t.height)" &&
  uv run --frozen python -m ggrpo.score --server.url $url --score.tasks $1 --score.predictions $2 \
    --score.out ${2%.jsonl}.scored.jsonl --score.reference results/grpo_eval/none.jsonl --log.file runs/ggrpo.jsonl \
    2>&1 | tail -n 1 | grep -oE '"verdicts.*'
}
sample() {  # sample PREDICTIONS: 8 completions per dev task, scored as one batch (verdict, passed; raw dropped)
  uv run --frozen python - $1 <<PY
import httpx, polars as pl, sys
p = pl.read_ndjson(sys.argv[1])
assert p.height == 8 * pl.read_ndjson("data/dev.jsonl").height and p.group_by("task_id").len()["len"].unique().to_list() == [8]
s = httpx.post("$url", json=p.select("task_id", completion="raw").to_dicts(), timeout=None).raise_for_status().json()
s = p.hstack(pl.DataFrame(s).select("verdict", "passed")).drop("raw")
s.write_ndjson(sys.argv[1].removesuffix(".jsonl") + ".scored.jsonl")
print("pass@1", round(s["passed"].mean(), 4), "pass@8", s.group_by("task_id").agg(pl.col("passed").any())["passed"].sum())
PY
}
report() {  # report FILE CMD...: run the scorer, log SCORED with its summary or SCORE_FAILED (retried next round)
  local f=$1 r; shift
  r=$("$@") && echo "SCORED $f $r $(date -u +%FT%TZ)" || echo "SCORE_FAILED $f $(date -u +%FT%TZ)"
}
echo "SCORE_START $name $(date -u +%FT%TZ)"
while :; do
  left=0
  for job in "$@"; do
    n=${job%:*} set=${job#*:}
    case $set in
      sample) f=$out/${n}_dev_sample8.jsonl ;;
      dev) f=$out/checkpoint-${n}_dev.jsonl ;;
      eval | split) f=$out/checkpoint-${n}_humaneval.jsonl ;;
    esac
    [ -f ${f%.jsonl}.scored.jsonl ] && continue
    left=1
    case $set in
      sample) ok totoro.mbzu.ae "SAMPLE_DONE $n 0 " ggrpo/runs/eval_gpu1_$name.status && scp -q totoro.mbzu.ae:$src/${n}_dev_sample8.jsonl $f &&
        report $f sample $f ;;
      dev) ok totoro.mbzu.ae "GEN_DONE $n dev 0 " ggrpo/runs/eval_gpu1_$name.status && scp -q totoro.mbzu.ae:$src/checkpoint-${n}_dev.jsonl $f &&
        report $f beam data/dev.jsonl $f ;;
      eval) ok ponyo "GEN_DONE $n eval 0 " gg/ggrpo/runs/eval_ponyo_$name.status && scp -q ponyo:gg/$src/checkpoint-${n}_eval.jsonl $f &&
        report $f beam data/eval.jsonl $f ;;
      split) ok totoro.mbzu.ae "GEN_DONE $n humaneval_gpu1 0 " ggrpo/runs/eval_gpu1_$name.status &&
        ok ponyo "GEN_DONE $n humaneval_ponyo 0 " gg/ggrpo/runs/eval_ponyo_$name.status &&
        cat <(ssh -o BatchMode=yes totoro.mbzu.ae "cat $src/checkpoint-${n}_humaneval_gpu1.jsonl") \
          <(ssh -o BatchMode=yes ponyo "cat gg/$src/checkpoint-${n}_humaneval_ponyo.jsonl") > $f.part && mv $f.part $f &&
        report $f beam data/eval.jsonl $f ;;
    esac
  done
  f=results/training/${out##*/}/trainer_state.json
  if [ ! -f $f ]; then
    left=1
    scp -q totoro.mbzu.ae:ggrpo/runs/$name/checkpoint-$final/trainer_state.json $f 2>/dev/null && echo "COPIED $f $(date -u +%FT%TZ)"
  fi
  [ $left = 0 ] && break
  sleep 120
done
echo "SCORE_DONE $name $(date -u +%FT%TZ)"
