import os, json
from tqdm import tqdm


src_dir = "evaluation/out/GSM8K/myllama-500M-multigpu"


model2score = {}
for model_id in os.listdir(src_dir):
    if "70" in model_id:
        continue
    model_dir = os.path.join(src_dir, model_id)
    model_scores = []
    for file in tqdm(os.listdir(model_dir)):
        if not file.startswith("ID"):
            continue

        with open(os.path.join(model_dir, file), "r") as f:
            data = json.load(f)

        score = data["orm_score"]
        model_scores.append(score)

    avg_score = sum(model_scores) / len(model_scores)
    model2score[model_id] = avg_score

print(model2score)

# Output:
res = {
    "tokens-10.00B-hf": -35.29965883244883,
    "tokens-20.00B-hf": -35.434893858984076,
    "tokens-40.00B-hf": -34.50094768764215,
    "tokens-50.00B-hf": -34.536959818043975,
    "tokens-60.00B-hf": -34.2778620166793,
}
