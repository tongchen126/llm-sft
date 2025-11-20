#!/bin/bash

MODEL=${MODEL:-"Qwen/Qwen3-VL-8B-Instruct"}
DATASET=${DATASET:-"pokemon1_cot2"}
HOST=${HOST:-"127.0.0.1"}
TRAIN_TYPE=${TRAIN_TYPE:-"lora"}
TRAIN_METHOD=${TRAIN_METHOD:-"rlhf"}
USE_CHORD=${USE_CHORD:-0}

DATASET_DIR="${WORKSPACE}/llm-sft/data/${DATASET}/"
SAVE_PATH="saved/${DATASET}_epoch${EPOCH}_${TRAIN_METHOD}/${MODEL##*/}/"

LORA_RANKS=(8) # 16 32)
LEARNING_RATES=(1e-5)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
nproc_per_node=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

export ROOT_IMAGE_DIR=${DATASET_DIR}
export NPROC_PER_NODE=$nproc_per_node
export LOG_LEVEL=INFO
export MS_SWIFT_LOG_LEVEL=INFO
export SWIFT_LOG_LEVEL=INFO
export WANDB_MODE=offline

echo "Model will be saved to $SAVE_PATH, using dataset ${DATASET_DIR}"
echo "Use GPU: ${CUDA_VISIBLE_DEVICES}, total number of GPU ${NPROC_PER_NODE}"

# ==============================================================================
# Static Parameters (Common across all runs)
# ==============================================================================
STATIC_PARAMS="
    --model ${MODEL} \
    --dataset ${DATASET_DIR}/data.json --split_dataset_ratio 0.01 \
    --num_train_epochs ${EPOCH} --eval_steps 100 --save_steps 100 --save_total_limit 5 --logging_steps 50 --warmup_ratio 0.1 --dataloader_num_workers 8 --dataset_num_proc 2 \
    --lr_scheduler_type cosine --load_from_cache_file true --gradient_checkpointing true --report_to all --use_hf true --torch_dtype bfloat16 \
    --deepspeed zero1 --max_length 4096 \
    --freeze_vit true --target_modules all-linear \
"

GRPO_PARAMS="
    --log_completions true --max_completion_length 2048
    --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 4 \
    --rlhf_type grpo --num_generations 8 --temperature 1.0 --beta 0.001 \
    --external_plugins /workspace/user_code/workspace_40172/llm-sft/external/grpo/pokemon.py --reward_funcs pokemon_grpo_format pokemon_grpo_acc repetition \
    --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.4 --vllm_tensor_parallel_size 1 --vllm_max_model_len 4096 --vllm_data_parallel_size ${nproc_per_node}
" 
# --val_dataset ${DATASET_DIR}/data_train_eval.json \
# --split_dataset_ratio 0.01 --dataset_num_proc 10
# --use_liger_kernel True --attn_impl flash_attn --log_completions true \
# --use_vllm true --vllm_mode server --vllm_server_host ${HOST} --vllm_server_port 8000
# --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.4 --vllm_tensor_parallel_size 1 --vllm_max_model_len 8192 --vllm_data_parallel_size 4

CHORD_PARAMS="
    --chord_sft_per_device_train_batch_size 1 \
    --chord_sft_dataset ${DATASET_DIR}/data.json \
    --chord_enable_phi_function false \
    --chord_mu_warmup_steps 25 \
    --chord_mu_decay_steps 300 \
    --chord_mu_peak 0.75 \
    --chord_mu_valley 0.15
"
if [ "$TRAIN_METHOD" == "rlhf" ]; then
	STATIC_PARAMS += $GRPO_PARAMS
fi

if [ $USE_CHORD -eq 1 ]; then
	STATIC_PARAMS += $CHORD_PARAMS
fi

# Iteration counter
iter_num=1

# Loop through all combinations
for LORA in "${LORA_RANKS[@]}"; do
    # Calculate lora_alpha (always 2 * LORA)
    LORA_ALPHA=$((2 * LORA))

    for LR in "${LEARNING_RATES[@]}"; do
        # Extract the exponent from learning rate (1e-4 -> 4, 1e-5 -> 5, etc.)
        LR_LOG=$(echo "$LR" | sed 's/1e-//')

        # Create the save name
        #SAVE_NAME="sft-${iter_num}-${LR_LOG}-r${LORA}-b1"
        SAVE_NAME="grpo-e${LR_LOG}-r${LORA}-b1${APPENDIX}"

        echo "================================================"
        echo "Running iteration ${iter_num}:"
        echo "  LORA rank: ${LORA}"
        echo "  LORA alpha: ${LORA_ALPHA}"
        echo "  Learning rate: ${LR}"
        echo "  Output: ${SAVE_PATH}/${SAVE_NAME}"
        echo "================================================"

        # Run the command
        echo "swift ${TRAIN_METHOD} ${STATIC_PARAMS} --train_type ${TRAIN_TYPE} --output_dir ${SAVE_PATH}/${SAVE_NAME} --learning_rate ${LR} --lora_rank ${LORA} --lora_alpha ${LORA_ALPHA}"
        swift ${TRAIN_METHOD} ${STATIC_PARAMS} --train_type ${TRAIN_TYPE} --output_dir ${SAVE_PATH}/${SAVE_NAME} --learning_rate ${LR} --lora_rank ${LORA} --lora_alpha ${LORA_ALPHA}

        # Increment iteration counter
        ((iter_num++))
    done
done
