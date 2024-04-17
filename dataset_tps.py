import gc
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from models import Naive_LLM, Naive_LLM_Quant, LLM_Flash
from datasets import Spider, DialogSum, E2ENLG, create_dataloader

import argparse

if __name__ == "__main__":
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="LLM_Flash")
    parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-2.7B")
    args = parser.parse_args()

    spider = Spider(download=True, train=False)
    spider_prompts = [sample.question for sample in spider]
    # print(spider_prompts[:5])

    dialogsum = DialogSum(download=True, train=False)
    dialogsum_prompts = [sample.summary for sample in dialogsum] # not actually the prompt, need to come up with one
    # print(dialogsum_prompts[:5])

    e2e_nlg = E2ENLG(download=True, train=False)
    e2e_nlg_prompts = [sample.ref for sample in e2e_nlg] # same, not actually the prompt
    # print(e2e_nlg_prompts[:5])

    
    print(f'Model_name : {args.model_name}, Model_Type: {args.model}')
    if args.model == "Naive_LLM":
        neo = Naive_LLM(args.model_name)
    elif args.model == "LLM_Quant":
        neo = Naive_LLM_Quant(args.model_name)
    elif args.model == "LLM_Flash":
        neo = LLM_Flash(args.model_name)
        
    neo.transformers.eval()
    config = GenerationConfig(max_new_tokens=256, pad_token_id=neo.generate_tokenizer.pad_token_id)

    total_tokens, total_time = 0, 0.0
    # for idx in tqdm(range(0, len(spider_prompts), batch_size)):
    for idx in tqdm(range(0, 10, batch_size)):
        _, tokens, elapsed = neo.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"Spider: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    # for idx in tqdm(range(0, len(dialogsum_prompts), batch_size)):
    for idx in tqdm(range(0, 10, batch_size)):
        _, tokens, elapsed = neo.generate(dialogsum_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"DialogSum: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    # for idx in tqdm(range(0, len(e2e_nlg_prompts), batch_size)):
    for idx in tqdm(range(0, 10, batch_size)):
        _, tokens, elapsed = neo.generate(e2e_nlg_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"E2E-NLG: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")
    
    

