checkpoint_dir=/n/holyscratch01/glassman_lab/Users/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-500M-multigpu/final
converted_dir=/n/holyscratch01/glassman_lab/Users/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-500M-multigpu/final-hf

litgpt convert_from_litgpt $checkpoint_dir $converted_dir
