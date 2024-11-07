from huggingface_hub import snapshot_download

model_id = "TinyLlama/TinyLlama_v1.1"

custom_download_path = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_models/hf_ckpts"

snapshot_path = snapshot_download(repo_id=model_id, local_dir=custom_download_path)

print(f"Model downloaded to: {snapshot_path}")
