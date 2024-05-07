import argparse
from tqdm import tqdm
import torch  
import wandb

from transformers import GenerationConfig


from models import Naive_LLM, Quant_LLM, LLM_Flash, LLM_MultiStream
from datasets import Spider, DialogSum, E2ENLG, create_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="naive")
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--use_wandb', type=bool, default=True)
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="HPML-6998")
    
    
    batch_size = 1

    spider = Spider(download=True, train=False)
    spider_prompts = [sample.question for sample in spider]

    dialogsum = DialogSum(download=True, train=False)
    dialogsum_prompts = [sample.summary for sample in dialogsum] # not actually the prompt, need to come up with one

    e2e_nlg = E2ENLG(download=True, train=True)
    e2e_nlg_prompts = [sample.ref for sample in e2e_nlg] # same, not actually the prompt

    print('GPT-Neo 2.7b:')
    if args.model_type == 'naive':
        print("LOADING NAIVE LLM")
        neo = Naive_LLM("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'quant':
        print("LOADING QUANTIZED LLM")
        neo = Quant_LLM("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'flash':
        print("LOADING FLASH LLM")
        neo = LLM_Flash("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'spec_stream':
        print("LOADING MULTI STREAM LLM")
        neo = LLM_MultiStream("EleutherAI/gpt-neo-2.7B")
    else:
        assert False, "Model Type not defined"
        
    neo.transformers.eval()
    config = GenerationConfig(max_new_tokens=256, pad_token_id=neo.generate_tokenizer.pad_token_id)
    spider_total_tokens, spider_total_time = 0, 0.0
    # for idx in tqdm(range(0, len(spider_prompts), batch_size)):
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        spider_total_tokens += tokens
        spider_total_time += elapsed
    
    print(f"Spider: Tokens/sec: {spider_total_tokens / spider_total_time} ({spider_total_time} seconds)")

    dialog_total_tokens, dialog_total_time = 0, 0.0
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(dialogsum_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        dialog_total_tokens += tokens
        dialog_total_time += elapsed
    print(f"DialogSum: Tokens/sec: {dialog_total_tokens / dialog_total_time} ({dialog_total_time} seconds)")

    e2e_total_tokens, e2e_total_time = 0, 0.0
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(e2e_nlg_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        e2e_total_tokens += tokens
        e2e_total_time += elapsed
    print(f"E2E-NLG: Tokens/sec: {e2e_total_tokens / e2e_total_time} ({e2e_total_time} seconds)")

    if args.use_wandb:
        wandb.log({"Spider": f"{round(spider_total_tokens / spider_total_time, 2)}tok/sec", "Dialog": f"{round(dialog_total_tokens / dialog_total_time,2)}tok/sec", "E2E":f"{round(e2e_total_tokens / e2e_total_time,2)}tok/sec"})