python evaluation/collect_responses.py \
    --prompt_template_file evaluation/prompts/fewshot_cot_template.txt \
    --prompt_config_file evaluation/prompts/fewshot_cot_config.json \
    --data_file evaluation/data/GSM8K/test_all.json \
    --model_ckpt_dir /n/holyscratch01/glassman_lab/Users/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-125M-multigpu/tokens-17.50B-hf \
    --out_dir evaluation/out/GSM8K/myllama-125M-multigpu/tokens-17.50B-hf \

