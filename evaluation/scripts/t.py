import os, json
from tqdm import tqdm


# src_dir = "evaluation/out/GSM8K/myllama-500M-multigpu"


# model2score = {}
# for model_id in os.listdir(src_dir):
#     model_dir = os.path.join(src_dir, model_id)
#     model_scores = []
#     for file in tqdm(os.listdir(model_dir)):
#         if not file.startswith("ID"):
#             continue

#         with open(os.path.join(model_dir, file), "r") as f:
#             data = json.load(f)

#         score = data["orm_score"]
#         model_scores.append(score)

#     avg_score = sum(model_scores) / len(model_scores)
#     model2score[model_id] = avg_score

# print(model2score)

"""
# Output
{
    "tokens-10.00B-hf": -35.29965883244883,
    "tokens-20.00B-hf": -35.434893858984076,
    "tokens-30.00B-hf": -34.85822592873389,
    "tokens-40.00B-hf": -34.50094768764215,
    "tokens-50.00B-hf": -34.536959818043975,
    "tokens-60.00B-hf": -34.2778620166793,
    "tokens-70.00B-hf": -34.15049279757392,
    "final-hf": -34.032031842304775,
}
"""

# Draw a line plot
import matplotlib.pyplot as plt
import seaborn as sns

model2score = {
    "10B": -35.29965883244883,
    "20B": -35.434893858984076,
    "30B": -34.85822592873389,
    "40B": -34.50094768764215,
    "50B": -34.536959818043975,
    "60B": -34.2778620166793,
    "70B": -34.15049279757392,
    "80B": -34.032031842304775,
}

plt.figure(figsize=(8, 8))
sns.lineplot(x=list(model2score.keys()), y=list(model2score.values()))
plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("ORM Score")
plt.title("Model vs ORM Score")
plt.savefig("evaluation/figs/model_vs_orm_score.png")
