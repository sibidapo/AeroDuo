#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/data/Hal-13k"
STAGE1_CKPT="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/high_uav/checkpoint/main/final/trainable_state.pt"
OUTPUT_DIR="checkpoint/stage2/main"

accelerate launch train.py \
  --dataset_root         "$DATASET_ROOT" \
  --stage1_ckpt          "$STAGE1_CKPT" \
  --output_dir           "$OUTPUT_DIR" \
  --batch_size           1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs     10 \
  --mixed_precision      bf16 \
  --learning_rate        3e-4 \
  --num_warmup_steps     200 \
  --checkpointing_steps  500 \
  --checkpoints_total_limit 3 \
  --window_T             5 \
  --action_horizon       8 \
  --seed                 42 \
  --wandb_project        aeroduo-stage2 \
  --wandb_run_name       stage2-run-1
