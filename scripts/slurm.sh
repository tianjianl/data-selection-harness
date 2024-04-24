#!/bin/bash

#SBATCH --job-name=llama3-oasst1-lora-64
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --mail-user=tli104@jh.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs_llama/llama3-oasst1-lora-64.log
#SBATCH --exclude=c001,c006

DATASET=$1

OUTPUT_DIR=/scratch/tli104/llama3_checkpoints/${DATASET}_llama3_8b_lora_1e4_r64

bash scripts/finetune_lora_with_accelerate.sh ${DATASET} ${OUTPUT_DIR}

eval "$(/home/tli104/miniconda3/bin/conda shell.bash hook)"

conda activate ftenv


python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/dolly-fullft-8B-0shot \
    --model_name_or_path /scratch/tli104/llama3_checkpoints/oasst1_llama3_8b_lora_1e4_r128_merged/ \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir /scratch/tli104/llama3_eval_results/${DATASET}_llama3_8b_lora_1e4_r64 \
    --model ${OUTPUT_DIR}_merged \
    --tokenizer meta-llama/Meta-Llama-3-8B \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format



