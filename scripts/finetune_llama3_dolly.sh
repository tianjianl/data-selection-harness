export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
   open_instruct/finetune.py \
    --model_name_or_path output/dolly_llama3_${MODEL_SIZE}_1e-5/ \
    --use_flash_attn \
    --tokenizer_name meta-llama/Meta-Llama-3-8B \
    --use_slow_tokenizer \
    --train_file data/processed/dolly/dolly_data.jsonl \
    --max_seq_length 8192 \
    --preprocessing_num_workers 30 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 4 \
    --output_dir output/dolly_llama3_${MODEL_SIZE}_1e-5/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --timeout 14400 
