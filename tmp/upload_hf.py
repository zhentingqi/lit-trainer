from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-1B-20BT-4gpus/final-hf",
    repo_id="ZhentingNLP/myllama-1B-20BT",
    repo_type="model",
    token="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",
)
