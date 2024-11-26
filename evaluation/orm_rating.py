import os, json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

orm_model = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_models/hf_ckpts/Skywork-Reward-Llama-3.1-8B-v0.2"
orm = AutoModelForSequenceClassification.from_pretrained(
    orm_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    num_labels=1,
)
orm_tokenizer = AutoTokenizer.from_pretrained(orm_model)


def get_chat_prompt(tokenizer, query, response):
    chat_prompt = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
    chat_prompt_tokenized = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
    return chat_prompt_tokenized


src_dir = "/n/home06/zhentingqi/lit-trainer/evaluation/out/GSM8K/myllama-500M-multigpu"

for model_id in os.listdir(src_dir):
    print(f"Processing {model_id}")
    model_dir = os.path.join(src_dir, model_id)
    if not os.path.isdir(model_dir):
        continue
    for file in tqdm(os.listdir(model_dir)):
        assert file.endswith(".json")
        if not file.startswith("ID"):
            continue
        
        file_path = os.path.join(model_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            
        if "orm_score" in data:
            continue
        
        id, problem, solution, generated = data["id"], data["problem"], data["solution"], data["generated"]
        prompt = get_chat_prompt(orm_tokenizer, problem, generated)
        inputs = orm_tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            score = orm(**inputs).logits.view(-1).cpu().float().item()
            
        data["orm_score"] = score
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
