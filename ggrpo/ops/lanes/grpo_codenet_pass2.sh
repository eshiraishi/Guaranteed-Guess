#!/usr/bin/env bash
# The second pass over the RL pool, chained after the first (Enzo, Nov 18: "Chain it now", once the measured 77 s/step
# made a second pass ~4 h rather than the ~2 days the memory probe's worst-batch rate implied).
#
# A continuation, not a new arm: same output_dir, same W&B run id, resumed from the first pass's final checkpoint, with
# max_steps raised from 184 to 368. The LR is constant_with_warmup (Nov 11), so step 185 trains at exactly the rate a
# single 368-step run would have used -- this is why the schedule was chosen constant in the first place. save_steps 46
# divides both 184 and 368, so the first pass ends on a full-state checkpoint and the second ends on one too.
#
# A separate script rather than an argument to grpo_codenet.sh: bash reads a script as it executes, so editing the
# running lane's file mid-run can corrupt its parse position.
# usage: grpo_codenet_pass2.sh SEED [ARM]   (ARM: grpo, the default, or rft)
set -uo pipefail
cd ~/ggrpo
seed=$1 arm=${2:-grpo}
case $arm in                            # the online-RFT control is the same lane with one module swapped (prereg.md: GRPO's sampler, LR and steps)
  grpo) module=ggrpo.record;   name=codenet_seed$seed; run="GRPO (datasets: mbpp+codenet)" ;;
  rft)  module=ggrpo.positive; name=rft_seed$seed;     run="Online RFT (datasets: mbpp+codenet)" ;;
  *)    echo "unknown arm $arm (grpo|rft)" >&2; exit 1 ;;
esac
steps=368
S=runs/${name}_pass2.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^GRPO_EXIT" runs/$name.status; do sleep 120; done
grep -q "^GRPO_EXIT 0" runs/$name.status || { echo "PASS1_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
# the first pass's last step must carry optimizer state, or this would silently restart from an earlier checkpoint
resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -s "$resume/optimizer.pt" ] || { echo "NO_OPTIMIZER $resume $(date -u +%FT%TZ)" >> $S; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
export WANDB_RUN_ID=$arm-codenet-seed$seed WANDB_RESUME=allow   # the same W&B run continues, so the curve is unbroken
set -a; . ./.env; set +a
echo "START $resume -> $steps $(date -u +%FT%TZ)" >> $S
.venv/bin/python -m $module --config grpo_codenet.yaml --seed $seed --output_dir runs/$name --run_name "$run" \
  --max_steps $steps --resume_from_checkpoint $resume >> runs/$name.log 2>&1
echo "PASS2_EXIT $? $(date -u +%FT%TZ)" >> $S
