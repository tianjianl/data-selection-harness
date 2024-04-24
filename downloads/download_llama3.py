import transformers
import torch
import os

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_id = "meta-llama/Llama-2-7b-hf"
#model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
  token=os.environ["HF_TOKEN"]
)
