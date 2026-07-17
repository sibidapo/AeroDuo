#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET_ROOT="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/data/Hal-13k"
# main3: rebalanced architecture (2026-07-14) — checkpoints in main2 and
# earlier are shape-incompatible (DiT 16L/1536d → 8L/768d, Perceiver widened,
# GraphEncoder 3 → 4 layers); do not point --resume at them.
OUTPUT_DIR="checkpoint/main3"
RESUME=""   # e.g. "checkpoint/main3/checkpoint-5000/trainable_state.pt"

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_processes 1 train.py \
  --dataset_root                "$DATASET_ROOT" \
  --output_dir                  "$OUTPUT_DIR" \
  ${RESUME:+--resume "$RESUME"} \
  --batch_size                  1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps             40000 \
  --mixed_precision             bf16 \
  --learning_rate                1e-4 \
  --num_warmup_steps            1000 \
  --checkpointing_steps         500 \
  --checkpoints_total_limit     3 \
  --window_T                    5 \
  --action_horizon              8 \
  --seed                        42 \
  --wandb_project               aeroduo-stage1 \
  --wandb_run_name              stage1-run-3 \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
