import os
from datasets import load_dataset
import json 

def valid_problem(problem: str):
    bad_words = ["Step:", "##", "feel free to ask", "Problem:", "more than happy", "I hope"]
    return not any(bad_word in problem for bad_word in bad_words)

# Source and target directories
src_dir = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct-2-unfiltered/data"
tgt_dir = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct-2-filtered"


os.makedirs(tgt_dir, exist_ok=True)
tgt_file = f"{tgt_dir}/data.jsonl"

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("parquet", data_files=f"{src_dir}/*.parquet", streaming=True)["train"]

unique_problems = set()

row = 0
with open(tgt_file, "w") as f:
    for example in dataset:
        print(f"Processing row {row}...")
        problem = example["problem"]
        generated_solution = example["generated_solution"]
        expected_answer = example["expected_answer"]
        problem_source = example["problem_source"]
        problem_id = f"row{row}-{problem_source}"

        if problem not in unique_problems and valid_problem(problem):
            unique_problems.add(problem)
            js = {
                "problem_id": problem_id,
                "problem": problem,
                "generated_solution": generated_solution,
                "expected_answer": expected_answer,
                "problem_source": problem_source,
            }
            f.write(json.dumps(js) + "\n")
    
        row += 1
    