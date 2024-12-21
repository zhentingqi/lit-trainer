from huggingface_hub import snapshot_download, hf_hub_download

# fineweb
# allow_patterns = ["sample/350BT/*"]

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     local_dir="/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/fineweb/sample-350BT",
#     repo_type="dataset",
#     force_download=True,
#     local_dir_use_symlinks=False,
#     resume_download=False,
#     allow_patterns=allow_patterns,
#     max_workers=64,
# )

# openwebmath
# allow_patterns = ["data/*"]

# snapshot_download(
#     repo_id="open-web-math/open-web-math",
#     local_dir="/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/openwebmath",
#     repo_type="dataset",
#     force_download=True,
#     local_dir_use_symlinks=False,
#     resume_download=False,
#     allow_patterns=allow_patterns,
#     max_workers=16,
# )

# open-hermes
# allow_patterns = ["data/train-*.parquet"]

# snapshot_download(
#     repo_id="nvidia/OpenMathInstruct-2",
#     local_dir="/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/OpenMathInstruct-2-unfiltered",
#     repo_type="dataset",
#     force_download=True,
#     local_dir_use_symlinks=False,
#     resume_download=False,
#     allow_patterns=allow_patterns,
#     max_workers=16,
# )

# ScaleQuest
snapshot_download(
    repo_id="dyyyyyyyy/ScaleQuest-Math",
    local_dir="/n/netscratch/glassman_lab/Lab/zhentingqi/big_data/hf_datasets/ScaleQuest-unfiltered",
    repo_type="dataset",
    force_download=True,
    local_dir_use_symlinks=False,
    resume_download=False,
    max_workers=16,
)
