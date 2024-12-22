root=/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-1B-160BT-4gpus

lit_ckpt_dir=$root/final
lit_convert_out_dir=$root/final-converted
hf_ckpt_dir=$root/final-hf

litgpt convert_from_litgpt $lit_ckpt_dir $lit_convert_out_dir

python scripts/make_hf_model.py \
    --lit_convert_out_dir $lit_convert_out_dir \
    --save_dir $hf_ckpt_dir \
    --disable_test_vllm