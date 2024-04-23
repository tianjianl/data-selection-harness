import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import jsonlines
import sys
import evaluate
from evaluate import logging

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

def compute(
    predictions, model, tokenizer, prefix="", batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
):

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

    # Tokenize input texts with prefix
    tokenized_texts = [pref + text for pref, text in zip(prefix, predictions)]

    encodings = tokenizer(
        tokenized_texts,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

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
        if add_start_token:
            shift_logits = shift_logits[:, len(tokenizer(prefix)) - 1 :]
            shift_labels = shift_labels[:, len(tokenizer(prefix)) - 1 :]
            shift_attention_mask_batch = shift_attention_mask_batch[:, len(tokenizer(prefix)) - 1 :]

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Initialize the Perplexity metric

# Open the JSONL file
with jsonlines.open(jsonl_file_path) as reader:
    # Iterate over each instance in the file
    for instance in reader:
        # Extract user messages (prefixes) and assistant responses (outputs)
        prefixes = [message["content"] for message in instance["messages"] if message["role"] == "user"]
        outputs = [message["content"] for message in instance["messages"] if message["role"] == "assistant"]

        # Calculate perplexities
        perplexity_results = compute(
            model=model,
			tokenizer=tokenizer,
            prefix=prefixes,
            predictions=outputs,
			device=device
        )
        
        # Append perplexity results for this instance to the list
        all_perplexity_results.append(perplexity_results)

# Print or use the perplexity results as needed
for idx, result in enumerate(all_perplexity_results):
    print(f"Instance {idx+1} - Mean Perplexity: {result['mean_perplexity']}")

