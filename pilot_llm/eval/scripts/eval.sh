#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"                  # pilot_llm/eval
PILOT_LLM_DIR="$(dirname "$EVAL_DIR")"               # pilot_llm
REPO_ROOT="$(dirname "$PILOT_LLM_DIR")"              # aeroduo

# eval.py imports dualuavpilot/config/vlnce_src/utils (eval), high_uav/low_uav
# (pilot_llm), and src.* (repo root); relative data/ paths resolve from the root.
cd "$REPO_ROOT"
export PYTHONPATH="$EVAL_DIR:$PILOT_LLM_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --no_zgraph: checkpoint from the standalone low-UAV run (low_uav/launch_train.sh
# exp2) — no z_graph conditioning; Stage 1 / SAM2 / GroundingDINO are never loaded,
# so no --stage1_ckpt is needed.
python -u pilot_llm/eval/eval.py \
    --run_type eval \
    --batchSize 1 \
    --maxWaypoints 50 \
    --steps_per_plan 2 \
    --gpu_id 0 \
    --device 0 \
    --simulator_tool_port 50000 \
    --no_zgraph \
    --stage2_ckpt ./pilot_llm/low_uav/checkpoint/exp2/main1/final/trainable_state.pt \
    --eval_save_path ./output_testrun \
    --dataset_path ./data/test_unseen_small.json
