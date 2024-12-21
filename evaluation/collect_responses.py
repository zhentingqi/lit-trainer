from argparse import ArgumentParser
import os
import json
import torch
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams


def generate(api, model, tokenizer, prompt: str, **gen_kwargs):
    if api == "hf":
        tokenized_input = tokenizer(prompt, return_tensors="pt")

        # Provide attention mask and handle padding
        input_ids = tokenized_input["input_ids"].to(model.device)
        attention_mask = tokenized_input["attention_mask"].to(model.device)
        
        assert input_ids.ndim == 2
        assert input_ids.shape[0] == 1
        outputs = model.generate(
            input_ids, 
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        assert len(outputs) == 1
        output = outputs[0]
        output = output[input_ids.shape[-1]:]   # Only decode the generated tokens
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        output_text = output_text.strip().strip("\n")
    elif api == "vllm":
        sampling_params = SamplingParams(
            max_new_tokens=gen_kwargs["max_new_tokens"],
            do_sample=gen_kwargs["do_sample"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            stop_tokens=gen_kwargs["stop_strings"],
        )
        output_text = model.generate(prompt, sampling_params)
        #todo
    
    return output_text
        

def main(args):
    #! Load prompt
    with open(args.prompt_template_file, "r") as f:
        prompt_template = f.read()
    assert "<<<instruction>>>" in prompt_template
    
    with open(args.prompt_config_file, "r") as f:
        prompt_config = json.load(f)
    assert "stop_tokens" in prompt_config      
    
    #! Load data
    with open(args.data_file, "r") as f:
        data = json.load(f)

    #! Load model and tokenizer
    if args.api == "hf":
        tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_dir, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(args.model_ckpt_dir)
        model = AutoModelForCausalLM.from_config(config).to(args.device)
        state_dict = torch.load(f"{args.model_ckpt_dir}/model.pth")
        model.load_state_dict(state_dict)
        
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,     
            "repetition_penalty": args.repetition_penalty,  
            "stop_strings": prompt_config["stop_tokens"],
        }
    elif args.api == "vllm":
        model = LLM(model=args.model_ckpt_dir, enable_prefix_caching=True)
        gen_kwargs = {
            "max_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "stop_tokens": prompt_config["stop_tokens"],
        }
    
    #! Evaluate
    print(f"==> Generating {len(data)} examples")
    for entry in tqdm(data):
        id, problem, solution = entry["id"], entry["problem"], entry["solution"]
        tgt_file = os.path.join(args.out_dir, f"ID-{id}.json")
        if os.path.exists(tgt_file):
            continue
        
        prompt = prompt_template.replace("<<<instruction>>>", problem)
        output_text = generate(args.api, model, tokenizer, prompt, **gen_kwargs)
        
        js = {
            "id": id,
            "problem": problem,
            "solution": solution,
            "generated": output_text,
        }
        with open(tgt_file, "w") as f:
            json.dump(js, f, indent=4)
        
        if args.verbose:
            print("-" * 50)
            print(f"==> Problem ID: {id}")
            print(f"==> Problem: {problem}")
            print(f"==> Solution: {solution}")
            print(f"==> Generated: {output_text}")
        
    #! Save
    args_file = os.path.join(args.out_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"==> Args saved to {args_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt_template_file", type=str, default=None)
    parser.add_argument("--prompt_config_file", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_ckpt_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--api", type=str, choices=["vllm", "hf"], default="vllm")
    
    # llm
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    
    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    main(args)

