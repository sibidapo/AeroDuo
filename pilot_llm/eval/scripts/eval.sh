#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL2_DIR="$(dirname "$SCRIPT_DIR")"                 # pilot_llm/eval2
PILOT_LLM_DIR="$(dirname "$EVAL2_DIR")"              # pilot_llm
REPO_ROOT="$(dirname "$PILOT_LLM_DIR")"             # aeroduo

# eval.py imports dualuavpilot/config/vlnce_src/utils (eval2), high_uav/low_uav
# (pilot_llm), and src.* (repo root); relative data/ paths resolve from the root.
cd "$REPO_ROOT"
export PYTHONPATH="$EVAL2_DIR:$PILOT_LLM_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -u pilot_llm/eval2/eval.py \
    --run_type eval \
    --batchSize 1 \
    --maxWaypoints 50 \
    --steps_per_plan 2 \
    --gpu_id 0 \
    --device 0 \
    --simulator_tool_port 50000 \
    --stage1_ckpt ./pilot_llm/high_uav/checkpoint/main/final/trainable_state.pt \
    --stage2_ckpt ./pilot_llm/low_uav/checkpoint/stage2/main/checkpoint-5500/trainable_state.pt \
    --eval_save_path ./output \
    --dataset_path ./data/test_unseen_new.json
