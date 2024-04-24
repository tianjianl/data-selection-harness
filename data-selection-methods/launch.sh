NUM_GPUS=8

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
./data-selection-methods/ppl_accelerate.py ./data/processed/dolly/dolly_data.jsonl
