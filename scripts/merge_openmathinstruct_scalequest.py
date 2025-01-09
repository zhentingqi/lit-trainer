import os
import json
from tqdm import tqdm
from random import shuffle


openmathinstruct_file = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct-2-filtered/data.jsonl"
scalequest_file = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/ScaleQuest-filtered/data.jsonl"
merged_file = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct2-ScaleQuest/data.jsonl"
os.makedirs(os.path.dirname(merged_file), exist_ok=True)

with open(openmathinstruct_file, "r") as f, open(scalequest_file, "r") as g, open(merged_file, "w") as h:
    print("Merging OpenMathInstruct-2 and ScaleQuest datasets...")
    openmathinstruct_data = []
    for line in tqdm(f.readlines()):
        data = json.loads(line)
        data["from_dataset"] = "OpenMathInstruct-2"
        openmathinstruct_data.append(data)
    scalequest_data = []
    for line in tqdm(g.readlines()):
        data = json.loads(line)
        data["from_dataset"] = "ScaleQuest"
        scalequest_data.append(data)
    all_data = openmathinstruct_data + scalequest_data
    shuffle(all_data)
    print("Writing merged data to file...")
    for data in tqdm(all_data):
        h.write(json.dumps(data) + "\n")
    
print(f"Wrote {len(all_data)} merged data to {merged_file}")