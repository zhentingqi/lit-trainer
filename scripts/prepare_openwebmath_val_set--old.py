from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer
import os


max_tokens = 500_000_000

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")

# Define paths
new_data_dir = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/openwebmath/eval_500M"
os.makedirs(new_data_dir, exist_ok=True)

# Load the original dataset as an IterableDataset
print("Loading the original dataset...")
"""
original_data_path = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/openwebmath/whole_data/*.parquet"
dataset = load_dataset(
    'parquet',
    data_files=original_data_path,
    split='train',
    streaming=True
)
"""
dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

# Initialize variables
total_tokens = 0
buffer = []
buffer_num_tokens = 0
buffer_size = 100_000_000
file_index = 0

# Iterate over the dataset and collect examples
for example in dataset:
    # Replace 'text' with the appropriate field name if different
    text = example['text']
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    
    # Check if adding this example exceeds the max token limit
    if total_tokens + num_tokens > max_tokens:
        break
    
    total_tokens += num_tokens
    print(f"Total tokens: {total_tokens}")
    buffer.append(example)
    buffer_num_tokens += num_tokens
    
    # Write buffer to a parquet file when it reaches the buffer size
    if buffer_num_tokens >= buffer_size:
        buffer_dataset = Dataset.from_list(buffer)
        output_file = os.path.join(new_data_dir, f'data_{file_index}.parquet')
        buffer_dataset.to_parquet(output_file)
        buffer = []
        buffer_num_tokens = 0
        file_index += 1

# Write any remaining examples in the buffer
if buffer:
    buffer_dataset = Dataset.from_list(buffer)
    output_file = os.path.join(new_data_dir, f'data_{file_index}.parquet')
    buffer_dataset.to_parquet(output_file)

print(f"Total tokens collected: {total_tokens}")

# Verify that the new dataset can be loaded as an IterableDataset
small_dataset = load_dataset(
    'parquet',
    data_files=os.path.join(new_data_dir, '*.parquet'),
    split='train',
    streaming=True
)

# Check if the loaded dataset is an IterableDataset
print(f"Is the dataset an IterableDataset? {isinstance(small_dataset, IterableDataset)}")
