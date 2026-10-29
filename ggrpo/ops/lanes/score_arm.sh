#!/usr/bin/env bash
# Runs on the Mac (nohup), one training run (runs/NAME): scores each eval file as soon as its lane reports it done, into
# the paths registered under [track.runs.*] in config.toml (OUT), through the VM's server as for seed 0. Dev beam 8 and
# sampled dev come from the GPU1 lane (eval_gpu1_arm.sh); HumanEval 106 is GPU1's and ponyo's halves concatenated and
# checked against data/eval.jsonl before scoring. After training exits, copies checkpoint-106's trainer_state.json.
# Already scored files are skipped, so a relaunch resumes. BringUp is scored by hand (VM harness, cluster shards).
# usage: score_arm.sh NAME OUT     e.g. score_arm.sh dsm_seed0 results/grpo_eval/dsm-seed0
cd "$(dirname "$0")/../.."
name=$1 out=$2 url=http://192.168.64.2:8765/score
src=ggrpo/runs/${name}_eval
mkdir -p $out results/training/${out##*/}
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
  for n in 106 27 54 81; do
    f=$out/${n}_dev_sample8.jsonl
    if [ ! -f ${f%.jsonl}.scored.jsonl ]; then
      left=1
      ok totoro.mbzu.ae "SAMPLE_DONE $n 0 " ggrpo/runs/eval_gpu1_$name.status && scp -q totoro.mbzu.ae:$src/${n}_dev_sample8.jsonl $f &&
        report $f sample $f
    fi
    f=$out/checkpoint-${n}_dev.jsonl
    if [ ! -f ${f%.jsonl}.scored.jsonl ]; then
      left=1
      ok totoro.mbzu.ae "GEN_DONE $n dev 0 " ggrpo/runs/eval_gpu1_$name.status && scp -q totoro.mbzu.ae:$src/checkpoint-${n}_dev.jsonl $f &&
        report $f beam data/dev.jsonl $f
    fi
  done
  f=$out/checkpoint-106_humaneval.jsonl
  if [ ! -f ${f%.jsonl}.scored.jsonl ]; then
    left=1
    ok totoro.mbzu.ae "GEN_DONE 106 humaneval_gpu1 0 " ggrpo/runs/eval_gpu1_$name.status &&
      ok ponyo "GEN_DONE 106 humaneval_ponyo 0 " gg/ggrpo/runs/eval_ponyo_$name.status &&
      cat <(ssh -o BatchMode=yes totoro.mbzu.ae "cat $src/checkpoint-106_humaneval_gpu1.jsonl") \
        <(ssh -o BatchMode=yes ponyo "cat gg/$src/checkpoint-106_humaneval_ponyo.jsonl") > $f.part && mv $f.part $f &&
      report $f beam data/eval.jsonl $f
  fi
  f=results/training/${out##*/}/trainer_state.json
  if [ ! -f $f ]; then
    left=1
    ok totoro.mbzu.ae "_EXIT 0 " ggrpo/runs/$name.status && scp -q totoro.mbzu.ae:ggrpo/runs/$name/checkpoint-106/trainer_state.json $f &&
      echo "COPIED $f $(date -u +%FT%TZ)"
  fi
  [ $left = 0 ] && break
  sleep 120
done
echo "SCORE_DONE $name $(date -u +%FT%TZ)"
