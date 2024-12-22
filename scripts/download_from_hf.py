from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZhentingNLP/myllama-1B-20BT",
    local_dir="???",
    repo_type="model",
    token="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",
    force_download=True,
    local_dir_use_symlinks=False,
    resume_download=True,
)
