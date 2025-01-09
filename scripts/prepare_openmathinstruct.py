import os
from datasets import load_dataset
import json 


def is_valid_problem(problem: str):
    bad_words = [
        "\nStep", "feel free to ask!", "Solution:", "Solve:", "Evaluate:", 
        "Therefore", "Thus", "So,", 
        "Write another problem", "Let's", "Another problem", 
        "Answer:", "The final answer", 
        "Inspired problem:", "##", "since", "this does not make sense", "answer is",
        "You are a helpful assistant.", "another problem", "inspired by this one", "new problem", "New Problem", "This problem", "I generated a problem", "original problem", "created a problem", "\n\n\n", "Problem: ", "None of these.", "Prove", 
        "Can I", "How should I", "Can you", "How should you", "How do I", "How do you",
        'ormath customers.4C.3_4.2.dynamic=1,"$ ? servings"', "theJustin Bon Secretary of train-------------", "\\aryh8r165100214018848&user0000716281559567", "feed<link nameleck_b.called, >. If thispert(p),<<user",
        " " * 20, "[[\n" * 4, "]]\n" * 4, "(i" * 4, "\\," * 4, "\\ \\& " * 4, "\\left|" * 4,
        "why", "We simplify the expression", "First, we", "Firstly, we", "<|start_header_id|>assistant",
    ]
    bool_a = not any([bad_word.lower() in problem.lower() for bad_word in bad_words])
    
    bool_b = 20 < len(problem) < 2000 
    
    bool_c = all(not problem.endswith(s) for s in [",", ",$", ",$$", "{", "[", "("])
    
    if "\\begin{asy}" in problem:
        bool_d = False
    else:
        if "[asy]" in problem:
            left_asy_cnt = problem.count("[asy]")
            right_asy_cnt = problem.count("[/asy]")
            bool_d = left_asy_cnt == right_asy_cnt
        else:
            bool_d = True
    
    if "[asy]" not in problem:
        bool_e = problem.count("\n\n") < 10
    else:
        bool_e = True
        
    bool_f = sum(problem.lower().count(s.lower()) for s in ["Find", "Determine", "Compute", "Calculate", "Express", "Enter", "What is", "How many", "How much", "How many", "How long", "How far", "How fast", "How many times", "How many more", "How many fewer", "How many less", "How many greater", "How many larger", "How many smaller",]) <= 1 and problem.count("?") <= 1

    return bool_a and bool_b and bool_c and bool_d and bool_e and bool_f


def is_valid_solution(solution: str):
    return "no solution" not in solution.lower()


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
        problem_id = f"OpenMathInstruct2--row{row}"

        if problem not in unique_problems and is_valid_problem(problem) and is_valid_solution(generated_solution):
            unique_problems.add(problem)
            js = {
                "problem_id": problem_id,
                "problem": problem,
                "generated_solution": generated_solution,
                "extra_info": {
                    "expected_answer": expected_answer,
                    "problem_source": problem_source,
                }
            }
            f.write(json.dumps(js) + "\n")
    
        row += 1

print(f"Wrote {len(unique_problems)} unique problems to {tgt_file}")