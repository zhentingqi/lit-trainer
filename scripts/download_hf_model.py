from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-2-7b-hf"

custom_download_path = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_models/hf_ckpts/Llama-2-7b-hf"

snapshot_path = snapshot_download(repo_id=model_id, local_dir=custom_download_path)

print(f"Model downloaded to: {snapshot_path}")
