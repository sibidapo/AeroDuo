OUTPUT_DIR=model_output
EXP_NAME=smolvlm_main
mkdir -p ./pilot_llm/${OUTPUT_DIR}/${EXP_NAME}
export CUDA_LAUNCH_BLOCKING=1

# Set TMPDIR to prevent /tmp cleanup issues during long training runs
export TMPDIR="./pilot_llm/${OUTPUT_DIR}/${EXP_NAME}/.tmp"
mkdir -p "$TMPDIR"

accelerate launch --main_process_port=40067 --config_file pilot_llm/default_deepspeed.yaml pilot_llm/train_pilot_llm_smolvlm.py \
    --model_name_or_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
    --mission="prob" \
    --output_dir ./pilot_llm/model_output/smolvlm_main \
    --mixed_precision "bf16" \
    --lora=True \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=200 \
    --max_train_steps=2001 \
    --checkpoints_total_limit=4 \
    --wandb_entity="sibidapo3-georgia-institute-of-technology" \
    --resume ./pilot_llm/model_output/smolvlm_main/checkpoint-400 \
    2>&1 | tee ./pilot_llm/model_output/smolvlm_main/train_resume.log
