from huggingface_hub import HfApi, create_repo


token = "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"
remote_repo_id = "ZhentingNLP/OpenMathInstruct2-ScaleQuest"
repo_type = "dataset"
local_folder = "/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct2-ScaleQuest/formatted_data"
private = True

url = create_repo(
    repo_id=remote_repo_id, 
    repo_type=repo_type,
    token=token,
    private=private,
)
print(f"Repo created at {url}")

api = HfApi()
api.upload_folder(
    folder_path=local_folder,
    repo_id=remote_repo_id,
    repo_type=repo_type,
    token=token,
)
