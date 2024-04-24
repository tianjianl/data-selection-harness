# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

:'
# Evaluating llama 7B model using 0 shot directly
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama3-8B-0shot \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit


# Evaluating llama 7B model using 5 shot directly
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama3-8B-5shot \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit
'

# Evaluating Tulu 7B model using 0 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/dolly-fullft-8B-0shot \
    --model_name_or_path ./output/dolly_llama3_8b_lora_merged/ \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

:'
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/dolly-fullft-8B-0shot \
    --model_name_or_path /home/tli104/data-selection-harness/output/dolly_llama3_8B_1e-5 \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating Tulu 7B model using 5 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/dolly-fullft-8B-5shot \
    --model_name_or_path /home/tli104/data-selection-harness/output/dolly_llama3_8B_1e-5 \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

'

# Evaluating llama2 chat model using 0-shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama3-inst-8B-0shot \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating llama2 chat model using 5-shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama2-chat-7B-5shot \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

:'
# Evaluating chatgpt using 0 shot
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/chatgpt-0shot/ \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20


# Evaluating chatgpt using 5 shot
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/chatgpt-5shot/ \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20


# Evaluating gpt4 using 0 shot
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/gpt4-0shot/ \
    --openai_engine "gpt-4-0314" \
    --n_instances 100 \
    --eval_batch_size 20


# Evaluating gpt4 using 5 shot
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/gpt4-5shot/ \
    --openai_engine "gpt-4-0314" \
    --n_instances 100 \
    --eval_batch_size 20
'
