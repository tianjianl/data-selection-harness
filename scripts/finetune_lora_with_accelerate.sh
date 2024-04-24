eval "$(/home/tli104/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/tli104/ftenv

export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_SIZE=8B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
DATASET=$1
OUTPUT_DIR=$2
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name meta-llama/Meta-Llama-3-8B \
    --use_slow_tokenizer \
    --train_file data/processed/${DATASET}/${DATASET}_data.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 4 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 5 &&
python open_instruct/merge_lora.py \
    --base_model_name_or_path meta-llama/Meta-Llama-3-8B \
    --lora_model_name_or_path ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}_merged \
    --save_tokenizer
