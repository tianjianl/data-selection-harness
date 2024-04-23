import sys
import jsonlines

from perplexity import Perplexity 

# Initialize list to store perplexity results
all_perplexity_results = []

# Path to the JSONL file
jsonl_file_path = sys.argv[1]

# Initialize the Perplexity metric
perplexity_metric = Perplexity()

# Open the JSONL file
with jsonlines.open(jsonl_file_path) as reader:
    # Iterate over each instance in the file
    for instance in reader:
        # Extract user messages (prefixes) and assistant responses (outputs)
        prefixes = [message["content"] for message in instance["messages"] if message["role"] == "user"]
        outputs = [message["content"] for message in instance["messages"] if message["role"] == "assistant"]

        # Calculate perplexities
        perplexity_results = perplexity_metric.compute(
            model_id='meta-llama/Meta-Llama-3-8B-Instruct',
            prefix=prefixes,
            predictions=outputs
        )
        
        # Append perplexity results for this instance to the list
        all_perplexity_results.append(perplexity_results)

# Print or use the perplexity results as needed
for idx, result in enumerate(all_perplexity_results):
    print(f"Instance {idx+1} - Mean Perplexity: {result['mean_perplexity']}")

