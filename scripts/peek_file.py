import json, os

file = "/n/holyscratch01/glassman_lab/Users/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-125M-multigpu/tokens-2.50B/config.json"

with open(file, 'r') as f:
    config = json.load(f)
    
print(json.dumps(config, indent=4))