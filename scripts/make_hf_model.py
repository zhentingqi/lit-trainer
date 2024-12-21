from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Trainer
import torch
from argparse import ArgumentParser
from vllm import LLM
import warnings
warnings.filterwarnings("ignore")


def save(args):
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

    trainer.save_model(args.save_dir)
    print(f"==> Model saved to {args.save_dir}")
    
    
def test(args):
    print(f"==> Testing model saved at {args.save_dir}")
    
    input_text = "Once upon a time"
    print(f"==> Input text: {input_text}")
    
    # Test tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir, use_fast=True)
    tokenized_input = tokenizer(input_text, return_tensors="pt").to("cuda")
    untokenized_input = tokenizer.decode(tokenized_input.input_ids[0])
    print(f"==> Tokenized-untokenized input: {untokenized_input}")
    
    # Test model
    model = LLM(model=args.save_dir)
    output = model.generate(prompts=[input_text])
    print(f"==> vllm output: {output[0]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lit_convert_out_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--disable_test_vllm", action="store_true")
    args = parser.parse_args()
    
    save(args)
    if not args.disable_test_vllm:
        test(args)
    