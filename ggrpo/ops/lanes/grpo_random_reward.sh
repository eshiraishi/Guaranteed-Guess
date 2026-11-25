#!/usr/bin/env bash
# The permuted-reward control, chained after the real run's second pass (Enzo, Nov 18: "chain the next most important
# ablation too").
#
# Why this one first of everything left: ggrpo.reward.reward_random takes each group's real rewards and shuffles them
# within the group, so the reward distribution is identical and only the pairing between a completion and its reward is
# destroyed. If this arm moves pass@1 as much as the real reward does, the headline result is an artefact of the RL
# machinery rather than of the verifier, and the paper has no H1 (Spurious Rewards, Shao et al., ICML 2026, found
# exactly that on several published setups). One flag, and it answers whether anything else is worth running.
#
# 184 steps, not 368: it is compared against the real arm's step-184 checkpoint, which is a full-state checkpoint we
# evaluate anyway. If 368 becomes the headline, this arm has to be extended to match before either goes in the paper.
# Same pool, same seed, same recipe, its own output_dir and W&B run.
# usage: grpo_random_reward.sh SEED
set -uo pipefail
cd ~/ggrpo
seed=$1 name=codenet_random_seed$1 real=codenet_seed$1
S=runs/$name.status
echo "WAIT $(date -u +%FT%TZ)" >> $S
until grep -q "^PASS2_EXIT" runs/${real}_pass2.status; do sleep 120; done
grep -q "^PASS2_EXIT 0" runs/${real}_pass2.status || { echo "REAL_RUN_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
export WANDB_RUN_ID=grpo-codenet-random-seed$seed WANDB_RESUME=allow
set -a; . ./.env; set +a
resume=$(ls -d runs/$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "START seed=$seed resume=${resume:-none} $(date -u +%FT%TZ)" >> $S
.venv/bin/python -m ggrpo.record --config grpo_codenet.yaml --seed $seed --output_dir runs/$name \
  --reward_funcs ggrpo.reward.reward_random \
  --run_name "GRPO (datasets: mbpp+codenet, reward_funcs: reward_random)" \
  ${resume:+--resume_from_checkpoint $resume} >> runs/$name.log 2>&1
echo "RANDOM_EXIT $? $(date -u +%FT%TZ)" >> $S
