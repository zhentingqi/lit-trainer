from huggingface_hub import snapshot_download

# fineweb
# allow_patterns = ["sample/350BT/*"]

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     local_dir="/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/fineweb/official",
#     repo_type="dataset",
#     force_download=True,
#     local_dir_use_symlinks=False,
#     resume_download=False,
#     allow_patterns=allow_patterns,
#     max_workers=16,
# )

# openwebmath
allow_patterns = ["data/*"]

snapshot_download(
    repo_id="open-web-math/open-web-math",
    local_dir="/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/openwebmath",
    repo_type="dataset",
    force_download=True,
    local_dir_use_symlinks=False,
    resume_download=False,
    allow_patterns=allow_patterns,
    max_workers=16,
)
