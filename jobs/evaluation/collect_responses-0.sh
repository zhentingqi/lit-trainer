python evaluation/collect_responses.py \
    --prompt_template_file evaluation/prompts/fewshot_cot_template.txt \
    --prompt_config_file evaluation/prompts/fewshot_cot_config.json \
    --data_file evaluation/data/GSM8K/test_all.json \
    --model_ckpt_dir /n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-1B-20BT-4gpus/final-hf \
    --out_dir evaluation/out/GSM8K/myllama-1B-20BT-4gpus/final-hf \

