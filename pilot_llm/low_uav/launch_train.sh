#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

##exp1 -> with z_graph
##exp2 -> without z_graph

TRAIN_DATA="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/data/train_data_new.json"
STAGE1_CKPT="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/high_uav/checkpoint/exp1/main1/final/trainable_state.pt"
OUTPUT_DIR="checkpoint/exp1/main1"
RESUME="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/low_uav/checkpoint/exp1/main1/checkpoint-16000/trainable_state.pt"

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 train.py \
  --stage1_ckpt                  "$STAGE1_CKPT" \
  --train_data                   "$TRAIN_DATA" \
  --output_dir                   "$OUTPUT_DIR" \
  ${RESUME:+--resume "$RESUME"} \
  --batch_size                   2 \
  --gradient_accumulation_steps  4 \
  --max_train_steps              50000 \
  --mixed_precision              bf16 \
  --learning_rate                1e-4 \
  --num_warmup_steps             1000 \
  --checkpointing_steps          2000 \
  --checkpoints_total_limit      5 \
  --window_T                     5 \
  --action_horizon               8 \
  --seed                         42 \
  --wandb_project                vanguard-stage2 \
  --wandb_run_name               stage2-w-zgraph-1 \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
