#!/usr/bin/env bash
# totoro GPU1 (4070 Ti SUPER), under runs/gpu1.lock: prereg E2 for the imitation arm's three LRs -- 16 samples per task
# over all 383 validation problems, at the training sampling settings (Enzo, Nov 19: "16, as prereg").
#
# One card, three runs back to back, not split across the two 4070s: ponyo has no vLLM in any of its venvs, and
# installing one there would put a different vLLM build on each half of the same comparison, which is a worse confound
# than the hours it saves. 6,128 completions per arm at the measured ~17 rows/min is about 6 h per arm, ~18 h for the
# grid.
#
# ggrpo.sample scores as it generates, so these rows come out with verdict and passed already set and there is no
# separate scoring step -- but the reward VM is busy for the whole run. The step-368 scoring lane wants the same VM for
# a few minutes; evals competing with evals is acceptable, idling a card for 18 h is not.
#
# Chained behind this card's step-368 beam half and behind the grid itself: an arm has to finish to be usable.
cd ~/ggrpo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=src TQDM_DISABLE=1
S=runs/e2_pool_refs.status
echo "WAIT grid and step-368 gpu1 half $(date -u +%FT%TZ)" >> $S
until grep -qE "^(GRID_EXIT|GRID_ABORT)" runs/sft_pool_references.status; do sleep 120; done
grep -q "^GRID_EXIT 0" runs/sft_pool_references.status \
  || { echo "GRID_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
until grep -qE "^(LANE_DONE|FAILED|SHORT|RECOVER_FAILED)" runs/codenet_val_grpo368/gpu1.status; do sleep 120; done
for lr in 5e-6 1e-5 2e-5; do
  arm=runs/sft_pool_refs_lr$lr
  [ -s $arm/model.safetensors ] || { echo "NO_WEIGHTS lr=$lr $(date -u +%FT%TZ)" >> $S; exit 1; }
  echo "E2_START lr=$lr $(date -u +%FT%TZ)" >> $S
  flock runs/gpu1.lock .venv/bin/python -m ggrpo.sample --sample.model $arm \
    --sample.dtype bfloat16 --sample.tasks runs/codenet_val_tasks_383.jsonl --sample.samples 16 \
    --sample.seed codenet-val-pool-refs-lr$lr --sample.stop_mixed 100000 \
    --sample.out runs/e2_pool_refs_lr${lr}_sample16.jsonl >> runs/e2_pool_refs_lr$lr.log 2>&1
  code=$?
  echo "E2_EXIT $code lr=$lr $(date -u +%FT%TZ)" >> $S
  # Nov 20: all three arms exited 1 and the lane still wrote E2_DONE, so the grid looked finished for 7.5 h
  # with no output file anywhere. A failed arm stops the lane now.
  [ "$code" = 0 ] || exit 1
done
echo "E2_DONE $(date -u +%FT%TZ)" >> $S
