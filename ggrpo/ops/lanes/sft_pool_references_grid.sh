#!/usr/bin/env bash
# The data-matched imitation arm's LR grid on GPU0, chained behind the permuted-reward control (Enzo, Nov 18: "Yes,
# record and run").
#
# SFT on the pool's compiler references: the same 2,944 tasks the GRPO run trains on, one reference each instead of a
# reward, from the same start (runs/sft_full_seed0). That is the comparison H1 is about -- without it the paper compares
# RL against nothing rather than against imitation on identical data.
#
# Three runs, 5e-6 / 1e-5 / 2e-5, one seed each, sequentially: GPU0 runs one job at a time, and an arm has to finish to
# be usable (Enzo, Nov 16). 2,944 pairs x 2 epochs / 16 per update = 368 updates, no generation, so each run is short
# next to a GRPO pass. The LR is then picked on val E2 and only the chosen one gets the confirmatory seeds; those evals
# are a separate step because scoring 16 samples x 383 tasks competes with training for the single reward VM.
#
# WANDB_API_KEY comes from .env, as sft_full.sh: without it a report_to [wandb] run dies in 38 s on "No API key
# configured". Run ids are readable (no plan IDs on W&B, Enzo, Nov 10), one per LR, so a relaunch continues that run.
set -uo pipefail
cd ~/ggrpo
S=runs/sft_pool_references.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^RANDOM_EXIT" runs/codenet_random_seed0.status; do sleep 120; done
grep -q "^RANDOM_EXIT 0" runs/codenet_random_seed0.status || { echo "CONTROL_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
[ -s data/sft/pool_references/train.parquet ] || { echo "no pool_references $(date -u +%FT%TZ)" >> $S; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 PYTHONPATH=src
set -a; . ./.env; set +a
for lr in 5e-6 1e-5 2e-5; do
  name=sft_pool_refs_lr$lr
  export WANDB_RUN_ID=sft-pool-refs-lr$lr WANDB_RESUME=allow
  echo "START lr=$lr $(date -u +%FT%TZ)" >> $S
  .venv/bin/python -m ggrpo.checkpointing --config sft_references.yaml \
    --dataset_name data/sft/pool_references --learning_rate $lr --output_dir runs/$name \
    --run_name "SFT on pool references (lr: $lr)" >> runs/$name.log 2>&1
  code=$?
  echo "ARM_EXIT $code lr=$lr $(date -u +%FT%TZ)" >> $S
  [ $code -eq 0 ] || { echo "GRID_ABORT lr=$lr $(date -u +%FT%TZ)" >> $S; exit 1; }
done
echo "GRID_EXIT 0 $(date -u +%FT%TZ)" >> $S
