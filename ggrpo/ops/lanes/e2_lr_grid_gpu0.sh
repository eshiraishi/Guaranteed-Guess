#!/usr/bin/env bash
# totoro GPU0 (Blackwell): prereg E2 for the imitation arm's three LRs -- 16 samples per task over all 383 validation
# problems, at the training sampling settings (Enzo, Nov 19: "16, as prereg"). It picks the LR of SFT on pool
# references, which the H1 comparator and offline RFT both wait on, so it is on the critical path and gets the fastest card
# (Enzo, Nov 20: the best GPU on the hardest work). The three arms are compared only with each other, so one card for all
# three keeps the one-GPU-type rule.
#
# Replaces e2_lr_grid_gpu1.sh, whose three arms all died on Nov 20 in ~60 s: TQDM_DISABLE=1 makes vLLM's progress
# bar report 0 s elapsed and its throughput line divides by it (ZeroDivisionError in offline_utils._run_engine). This
# lane leaves tqdm on. Nov 20, 19:20Z: restarted as two shards per model (e2_two_shards.sh); sample.py resumes from the
# rows already scored.
set -uo pipefail
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src
S=runs/e2_pool_refs.status
echo "RESTART gpu0 $(date -u +%FT%TZ)" >> $S
for lr in 5e-6 1e-5 2e-5; do
  arm=runs/sft_pool_refs_lr$lr
  [ -s $arm/model.safetensors ] || { echo "NO_WEIGHTS lr=$lr $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "E2_START lr=$lr gpu0 $(date -u +%FT%TZ)" >> $S
  bash runs/e2_two_shards.sh $arm codenet-val-pool-refs-lr$lr runs/e2_pool_refs_lr${lr}_sample16.jsonl \
    runs/e2_pool_refs_lr$lr.log
  code=$?
  echo "E2_EXIT $code lr=$lr $(date -u +%FT%TZ)" >> $S
  [ "$code" = 0 ] || exit 1
done
echo "E2_DONE $(date -u +%FT%TZ)" >> $S
