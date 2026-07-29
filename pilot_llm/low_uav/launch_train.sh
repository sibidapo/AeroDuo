#!/usr/bin/env bash
set -euo pipefail

##exp1 -> with z_graph
##exp2 -> without z_graph

TRAIN_DATA="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/data/train_data_new.json"
#STAGE1_CKPT="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/high_uav/checkpoint/main/final/trainable_state.pt"
OUTPUT_DIR="checkpoint/exp2/main1"

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 train.py \
  --no_zgraph \
  --resume               "$OUTPUT_DIR/checkpoint-26000/trainable_state.pt" \
  --train_data           "$TRAIN_DATA" \
  --output_dir           "$OUTPUT_DIR" \
  --batch_size           16 \
  --gradient_accumulation_steps 2 \
  --max_train_steps      40000 \
  --mixed_precision      bf16 \
  --learning_rate        1e-4 \
  --num_warmup_steps     1000 \
  --checkpointing_steps  2000 \
  --checkpoints_total_limit 5 \
  --window_T             1 \
  --action_horizon       8 \
  --seed                 42 \
  --wandb_project        vanguard-stage2 \
  --wandb_run_name       stage2-no-zgraph-1 \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"


# accelerate launch --num_processes 1 train.py \
#   --train_data           "$TRAIN_DATA" \
#   --stage1_ckpt          "$STAGE1_CKPT" \
#   --output_dir           "$OUTPUT_DIR" \
#   --batch_size           1 \
#   --gradient_accumulation_steps 4 \
#   --num_train_epochs     2 \
#   --mixed_precision      bf16 \
#   --learning_rate        3e-4 \
#   --num_warmup_steps     200 \
#   --checkpointing_steps  500 \
#   --checkpoints_total_limit 3 \
#   --window_T             5 \
#   --action_horizon       8 \
#   --seed                 42 \
#   --wandb_project        aeroduo-stage2 \
#   --wandb_run_name       stage2-run-2
