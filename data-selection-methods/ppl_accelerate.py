import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import jsonlines
import sys
import evaluate
from evaluate import logging

from accelerate import Accelerator

from torch.utils.data import DataLoader, Dataset

class InstructDataset(Dataset):
    def __init__(self, jsonl_file_path):
        self.jsonl_file_path = jsonl_file_path
        self.data = []
        # data looks like
        # {"dataset": "dolly", "id": "dolly_8", "messages": [{"role": "user", "content": "Why mobile is bad for human\n"}, {"role": "assistant", "content": "We are always engaged one phone which is not good."}]}
        # need to extract the user content as "prefix" and assistant content as "predictions"
        with jsonlines.open(jsonl_file_path) as reader:
            for obj in reader:
                prefix = obj["messages"][0]["content"]
                predictions = obj["messages"][1]["content"]
                self.data.append((prefix, predictions))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prefix, predictions = self.data[idx]
        return prefix, predictions
    
    
_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.
"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity.
        NOTE: Perplexity can only be calculated for causal language models.
        This includes models such as GPT-2, causal variations of BERT,
        causal versions of T5, and more (the full list can be found
        in the AutoModelForCausalLM documentation).
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    prefix (str): prefix to prepend to each input text for evaluation.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
"""

def compute(model, tokenizer, dataloader, device='cuda', batch_size=16, add_start_token=True, max_length=None, accelerator=None):
    
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length
    
    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for batch in dataloader:

        tokenized_batch = tokenizer(
            [prefix + prediction for prefix, prediction in batch],
            padding="max_length",
            max_length=max_tokenized_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoded_batch = tokenized_batch["input_ids"]
        attn_mask = tokenized_batch["attention_mask"]
        
        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # Exclude tokens corresponding to the prefix when computing loss
        for index, (prefix, _) in enumerate(batch):
            shift_logits = shift_logits[index, len(tokenizer(prefix)) - 1 :]
            shift_labels = shift_labels[index, len(tokenizer(prefix)) - 1 :]
            shift_attention_mask_batch = shift_attention_mask_batch[index, len(tokenizer(prefix)) - 1 :]

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

# Initialize list to store perplexity results
all_perplexity_results = []

# Path to the JSONL file
jsonl_file_path = sys.argv[1]
print(f"now evaluating perplexity of {jsonl_file_path}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = InstructDataset(jsonl_file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# Compute perplexity
perplexity_results = compute(
    model=model,
    tokenizer=tokenizer,
    dataloader=dataloader,
    device=device,
    batch_size=2,
    add_start_token=True,
    max_length=None,
)
