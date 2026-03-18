OUTPUT_DIR=model_output
EXP_NAME=smolvlm_main
mkdir ./pilot_llm/${OUTPUT_DIR}/${EXP_NAME}
export CUDA_LAUNCH_BLOCKING=1

accelerate launch --main_process_port=40067 --config_file pilot_llm/default_deepspeed.yaml pilot_llm/train_pilot_llm_smolvlm.py \
    --model_name_or_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
    --mission="prob" \
    --output_dir ${OUTPUT_DIR}/${EXP_NAME} \
    --mixed_precision "bf16" \
    --lora=True \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --checkpointing_steps=1000 \
    --max_train_steps=2001 \
    --checkpoints_total_limit=4 \
    2>&1 | tee ./pilot_llm/${OUTPUT_DIR}/${EXP_NAME}/train.log