import os
from datasets import load_dataset
import json

def valid_problem(problem: str):
    bad_words = ["Step:", "##", "feel free to ask", "Problem:", "more than happy", "I hope"]
    return not any(bad_word in problem for bad_word in bad_words)

# Source and target directories
src_file = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/ScaleQuest-unfiltered/train.json"
tgt_dir = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/ScaleQuest-filtered"
os.makedirs(tgt_dir, exist_ok=True)
tgt_file = f"{tgt_dir}/data.jsonl"

# Load the dataset
unique_problems = set()

row = 0
with open(tgt_file, "w") as tgt, open(src_file, "r") as src:
    for line in src.readlines():
        example = json.loads(line)
        print(f"Processing row {row}...")
        query = example["query"]
        response = example["response"]
        problem_id = f"row{row}"

        if query not in unique_problems and valid_problem(query):
            unique_problems.add(query)
            js = {
                "problem_id": problem_id,
                "problem": query,
                "generated_solution": response,
            }
            tgt.write(json.dumps(js) + "\n")
    
        row += 1
    
print(f"Wrote {len(unique_problems)} unique problems to {tgt_file}")