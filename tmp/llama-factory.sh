#!/bin/bash

MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct"
TEMPLATE="qwen3_vl_nothink"
DATASET_DIR="${WORKSPACE}/llm-sft/data/pokemon_label"
SAVE_PATH="saved/pokemon_label_epoch${EPOCH}/qwen3vl-30bA3b/"
echo $SAVE_PATH
# ==============================================================================
# Static Parameters (Common across all runs)
# ==============================================================================
STATIC_PARAMS="
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --lora_target all \
    --dataset pokemon \
    --dataset_dir ${DATASET_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len 2048 \
    --overwrite_cache true \
    --preprocessing_num_workers 20 \
    --dataloader_num_workers 20 \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs ${EPOCH} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000
"

llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-4 --output_dir ${SAVE_PATH}/sft-1-e4-r8-b1 		--finetuning_type lora --lora_rank 8
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-5 --output_dir ${SAVE_PATH}/sft-2-e5-r8-b1 		--finetuning_type lora --lora_rank 8
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-3 --output_dir ${SAVE_PATH}/sft-3-e3-r8-b1 	 	--finetuning_type lora --lora_rank 8
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-4 --output_dir ${SAVE_PATH}/sft-4-e4-r16-b1  	--finetuning_type lora --lora_rank 16
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-5 --output_dir ${SAVE_PATH}/sft-5-e5-r16-b1  	--finetuning_type lora --lora_rank 16
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-3 --output_dir ${SAVE_PATH}/sft-6-e3-r16-b1  	--finetuning_type lora --lora_rank 16
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-4 --output_dir ${SAVE_PATH}/sft-7-e4-r32-b1  	--finetuning_type lora --lora_rank 32
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-5 --output_dir ${SAVE_PATH}/sft-8-e5-r32-b1  	--finetuning_type lora --lora_rank 32
llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-3 --output_dir ${SAVE_PATH}/sft-9-e3-r32-b1  	--finetuning_type lora --lora_rank 32
#llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-4 --output_dir ${SAVE_PATH}/sft-7-e4-full-b1 	--finetuning_type full
#llamafactory-cli train ${STATIC_PARAMS} --learning_rate 1e-5 --output_dir ${SAVE_PATH}/sft-8-e5-full-b1 	--finetuning_type full
