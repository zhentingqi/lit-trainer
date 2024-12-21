from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Trainer
import torch
from argparse import ArgumentParser


def main(args):
    args.lit_convert_out_dir = "/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-1B-20BT-4gpus/final-hf"

    tokenizer = AutoTokenizer.from_pretrained(args.lit_convert_out_dir, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.lit_convert_out_dir)
    model = AutoModelForCausalLM.from_config(config).to("cuda")
    state_dict = torch.load(f"{args.lit_convert_out_dir}/model.pth")
    model.load_state_dict(state_dict)


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    save_dir = "/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/overtraining/lit_out/pretrain/myllama-1B-20BT-4gpus/final-hf-test"
    trainer.save_model(save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lit_convert_out_dir", type=str, required=True)
    args = parser.parse_args()