#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

##exp1 -> with z_graph
##exp2 -> without z_graph

TRAIN_DATA="/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/data/train_data_new.json"
OUTPUT_DIR="checkpoint/exp1/main1"
#RESUME="checkpoint/exp1/main1/checkpoint-2000/trainable_state.pt"
RESUME=""

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_processes 3 --num_machines 1 --mixed_precision bf16 train.py \
  --train_data                  "$TRAIN_DATA" \
  --output_dir                  "$OUTPUT_DIR" \
  ${RESUME:+--resume "$RESUME"} \
  --batch_size                  2 \
  --gradient_accumulation_steps 4 \
  --max_train_steps             40000 \
  --mixed_precision             bf16 \
  --learning_rate                1e-4 \
  --num_warmup_steps            1000 \
  --checkpointing_steps         1000 \
  --checkpoints_total_limit     5 \
  --window_T                    5 \
  --action_horizon              8 \
  --seed                        42 \
  --wandb_project               vanguard-stage1 \
  --wandb_run_name              stage1-run-1 \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"