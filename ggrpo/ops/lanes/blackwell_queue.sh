#!/usr/bin/env bash
# totoro GPU0 (Blackwell), the Nov 20 queue (Enzo: approved, "as long as we prioritize in the correct order").
# Order = what unblocks the most first, then H1 before what H1 gates:
#   1. wait for E2 of the imitation arm's LR grid (e2_lr_grid_gpu0.sh);
#   2. pick that LR as prereg says: the highest mean pass rate over the 16 x 383 val samples;
#   3. SFT on pool references, seeds 1-4 at that LR (seed 0 is the grid run), ~24 min each -- the H1 comparator;
#   4. offline RFT, 1 seed, same LR;
#   5. the GRPO LR check: 5e-7 and 2e-6, seed 0, one pass (184 steps), otherwise grpo_codenet.yaml;
#   6. E2 at step 184 for both LR-check runs and both 1e-6 seeds, all through e2_two_shards.sh on this card;
#   7. the GRPO LR, by the rule Enzo approved on Nov 20: 1e-6 stays unless the better LR-check run beats 1e-6 seed 0
#      (same seed) by more than |seed 0 - seed 1| at 1e-6. The LR check is reported as a sensitivity analysis either way;
#   8. the online seeds at that LR, two passes each (184 + 184 steps, as every online run so far), in priority order:
#      GRPO (H1's arm), then online RFT (H2), then the permuted-reward control. At 1e-6 the finished runs count; at
#      another LR they are re-run under new names, and GRPO seed 0 continues from its LR-check run.
#   Added Nov 21 (Enzo: "Ok"): after 4, a 2-step smoke test of each GRPO variant, then SFT on RL references (H2a,
#   prereg: LR grid on val E2, then 5 seeds), which had dropped out of the queue with the Nov 16 scope cut; after 8, the
#   GRPO variant baselines (not in prereg: descriptive, 1 seed each), each with its own 3-LR check at 184 steps, the
#   highest val E2 wins and that run continues to 368. DeepSeekMath GRPO (loss_type grpo, beta 0.04) uses GRPO's grid.
#   Dr. GRPO (loss_type dr_grpo, scale_rewards none) divides the loss by the 16,384-token cap instead of the ~1.5k tokens
#   a completion has, so its updates are ~10x smaller at the same LR: its grid is moved up to {1e-6, 3e-6, 1e-5}.
set -uo pipefail
cd ~/ggrpo
S=runs/blackwell_queue.status
log() { echo "$* $(date -u +%FT%TZ)" >> $S; }
log "WAIT e2"
e2log() { sed -n '/^RESTART gpu0/,$p' runs/e2_pool_refs.status; }
until e2log | grep -qE "^E2_(DONE|EXIT [^0])"; do sleep 120; done
e2log | grep -q "^E2_DONE" || { log E2_FAILED; exit 1; }
rate() { .venv/bin/python -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
assert len(rows) == 6128, len(rows)
print(sum(r['passed'] for r in rows) / len(rows))" $1; }
best=none bestrate=-1
for lr in 5e-6 1e-5 2e-5; do
  r=$(rate runs/e2_pool_refs_lr${lr}_sample16.jsonl) || { log "RATE_FAILED sft lr=$lr"; exit 1; }
  log "E2 sft_pool_refs lr=$lr pass=$r"
  if .venv/bin/python -c "import sys; sys.exit(not float(sys.argv[1]) > float(sys.argv[2]))" $r $bestrate; then best=$lr bestrate=$r; fi
done
lr=$best; log "CHOSEN_SFT_LR $lr"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 PYTHONPATH=src VLLM_USE_FLASHINFER_SAMPLER=0
set -a; . ./.env; set +a
run_sft() {  # name dataset seed run_name [lr]
  [ -s runs/$1/model.safetensors ] && grep -q "^EXIT 0 $1 " $S && return 0
  export WANDB_RUN_ID=$(echo $1 | tr _ -) WANDB_RESUME=allow
  log "START $1"
  .venv/bin/python -m ggrpo.checkpointing --config sft_references.yaml --dataset_name $2 --learning_rate ${5:-$lr} \
    --seed $3 --output_dir runs/$1 --run_name "$4" >> runs/$1.log 2>&1
  code=$?; log "EXIT $code $1"; [ $code -eq 0 ] || exit 1
}
for seed in 1 2 3 4; do
  run_sft sft_pool_refs_lr${lr}_seed$seed data/sft/pool_references $seed "SFT on pool references (lr: $lr, seed: $seed)"
done
run_sft offline_rft_lr${lr}_seed0 data/sft/offline_rft 0 "Offline RFT (lr: $lr)"
e2() {  # model salt out log
  [ "$(wc -l < $3 2>/dev/null || echo 0)" -eq 6128 ] && return 0
  log "E2_START $1"
  bash runs/e2_two_shards.sh $1 $2 $3 $4
  local code=$?; log "E2_EXIT $code $1"; [ $code -eq 0 ] || exit 1
}
for variant in dsm drgrpo; do   # a failure is logged and that variant is skipped at the end; the queue goes on
  grep -q "^SMOKE_OK $variant " $S && continue
  case $variant in dsm) extra="--loss_type grpo --beta 0.04" ;; drgrpo) extra="--loss_type dr_grpo --scale_rewards none" ;; esac
  rm -rf runs/smoke_$variant
  WANDB_MODE=disabled .venv/bin/python -m ggrpo.record --config grpo_codenet.yaml --seed 0 $extra --max_steps 2 \
    --save_strategy no --report_to none --output_dir runs/smoke_$variant --run_name smoke >> runs/smoke_$variant.log 2>&1 \
    && log "SMOKE_OK $variant" || log "SMOKE_FAILED $variant"
done
for rlr in 5e-6 1e-5 2e-5; do
  run_sft sft_refs_lr$rlr data/sft/references 0 "SFT on RL references (lr: $rlr)" $rlr
done
best=none bestrate=-1
for rlr in 5e-6 1e-5 2e-5; do
  e2 runs/sft_refs_lr$rlr codenet-val-refs-lr$rlr runs/e2_refs_lr${rlr}_sample16.jsonl runs/e2_refs_lr$rlr.log
  r=$(rate runs/e2_refs_lr${rlr}_sample16.jsonl) || { log "RATE_FAILED refs lr=$rlr"; exit 1; }
  log "E2 sft_refs lr=$rlr pass=$r"
  if .venv/bin/python -c "import sys; sys.exit(not float(sys.argv[1]) > float(sys.argv[2]))" $r $bestrate; then best=$rlr bestrate=$r; fi
done
rlr=$best; log "CHOSEN_REFS_LR $rlr"
for seed in 1 2 3 4; do
  run_sft sft_refs_lr${rlr}_seed$seed data/sft/references $seed "SFT on RL references (lr: $rlr, seed: $seed)" $rlr
done
online() {  # arm seed lr name wandb_id steps run_name
  local arm=$1 seed=$2 glr=$3 name=$4 id=$5 steps=$6 title=$7 module extra=
  case $arm in
    grpo) module=ggrpo.record ;;
    rft) module=ggrpo.positive ;;
    random) module=ggrpo.record; extra="--reward_funcs ggrpo.reward.reward_random" ;;
    dsm) module=ggrpo.record; extra="--loss_type grpo --beta 0.04" ;;
    drgrpo) module=ggrpo.record; extra="--loss_type dr_grpo --scale_rewards none" ;;
  esac
  local done_steps=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  [ "${done_steps:-0}" -ge $steps ] && { log "SKIP $name already at ${done_steps}"; return 0; }
  local resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -z "$resume" ] || [ -s "$resume/optimizer.pt" ] || { log "NO_OPTIMIZER $resume"; exit 1; }
  export WANDB_RUN_ID=$id WANDB_RESUME=allow
  log "START $name to=$steps resume=${resume:-none}"
  .venv/bin/python -m $module --config grpo_codenet.yaml --seed $seed --learning_rate $glr --max_steps $steps \
    --output_dir runs/$name --run_name "$title" $extra ${resume:+--resume_from_checkpoint $resume} >> runs/$name.log 2>&1
  local code=$?; log "EXIT $code $name to=$steps"; [ $code -eq 0 ] || exit 1
}
for glr in 5e-7 2e-6; do
  online grpo 0 $glr codenet_lr${glr}_seed0 grpo-codenet-lr$glr-seed0 184 "GRPO (datasets: mbpp+codenet, lr: $glr)"
done
for model in codenet_seed0 codenet_seed1 codenet_lr5e-7_seed0 codenet_lr2e-6_seed0; do
  e2 runs/$model/checkpoint-184 codenet-val-grpo-lr-$model runs/e2_grpo_lr_${model}_184.jsonl runs/e2_grpo_lr_$model.log
done
for model in codenet_seed0 codenet_seed1 codenet_lr5e-7_seed0 codenet_lr2e-6_seed0; do
  declare "r_${model//[-.]/_}=$(rate runs/e2_grpo_lr_${model}_184.jsonl)" || { log "RATE_FAILED $model"; exit 1; }
done
glr=$(.venv/bin/python - $r_codenet_seed0 $r_codenet_seed1 $r_codenet_lr5e_7_seed0 $r_codenet_lr2e_6_seed0 <<'PY'
import sys
seed0, seed1, low, high = map(float, sys.argv[1:])
noise = abs(seed0 - seed1)
alt, rate = max([("5e-7", low), ("2e-6", high)], key=lambda pair: pair[1])
print(alt if rate - seed0 > noise else "1e-6")
PY
) || { log "RULE_FAILED"; exit 1; }
log "E2 grpo@184 lr=1e-6 seed0=$r_codenet_seed0 seed1=$r_codenet_seed1 lr=5e-7 $r_codenet_lr5e_7_seed0 lr=2e-6 $r_codenet_lr2e_6_seed0"
log "CHOSEN_GRPO_LR $glr"
if [ "$glr" = 1e-6 ]; then tag= grpo_seeds="2 3 4" rft_seeds="1 2 3 4" random_seeds="0 1 2"
else tag=_lr$glr grpo_seeds="0 1 2 3 4" rft_seeds="0 1 2 3 4" random_seeds="0 1 2"; fi
idtag=${tag//_/-}
for seed in $grpo_seeds; do for steps in 184 368; do
  online grpo $seed $glr codenet${tag}_seed$seed grpo-codenet$idtag-seed$seed $steps "GRPO (datasets: mbpp+codenet${tag:+, lr: $glr})"
done; done
for seed in $rft_seeds; do for steps in 184 368; do
  online rft $seed $glr rft${tag}_seed$seed rft-codenet$idtag-seed$seed $steps "Online RFT (datasets: mbpp+codenet${tag:+, lr: $glr})"
done; done
for seed in $random_seeds; do for steps in 184 368; do
  online random $seed $glr codenet_random${tag}_seed$seed grpo-codenet-random$idtag-seed$seed $steps \
    "GRPO (datasets: mbpp+codenet, reward_funcs: reward_random${tag:+, lr: $glr})"
done; done
for variant in dsm drgrpo; do
  grep -q "^SMOKE_OK $variant " $S || { log "VARIANT_SKIPPED $variant (smoke test failed)"; continue; }
  case $variant in
    dsm) grid="5e-7 1e-6 2e-6" title="GRPO (DeepSeekMath: loss_type grpo, beta 0.04" ;;
    drgrpo) grid="1e-6 3e-6 1e-5" title="Dr. GRPO (loss_type dr_grpo, scale_rewards none" ;;
  esac
  best=none bestrate=-1
  for vlr in $grid; do
    online $variant 0 $vlr grpo_${variant}_lr${vlr}_seed0 grpo-$variant-lr$vlr-seed0 184 "$title, lr: $vlr)"
    e2 runs/grpo_${variant}_lr${vlr}_seed0/checkpoint-184 codenet-val-$variant-lr$vlr \
      runs/e2_${variant}_lr${vlr}_184.jsonl runs/e2_${variant}_lr$vlr.log
    r=$(rate runs/e2_${variant}_lr${vlr}_184.jsonl) || { log "RATE_FAILED $variant lr=$vlr"; exit 1; }
    log "E2 $variant@184 lr=$vlr pass=$r"
    if .venv/bin/python -c "import sys; sys.exit(not float(sys.argv[1]) > float(sys.argv[2]))" $r $bestrate; then best=$vlr bestrate=$r; fi
  done
  log "CHOSEN_LR $variant $best"
  online $variant 0 $best grpo_${variant}_lr${best}_seed0 grpo-$variant-lr$best-seed0 368 "$title, lr: $best)"
done
log QUEUE_DONE
