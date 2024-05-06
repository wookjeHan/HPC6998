import argparse
from tqdm import tqdm
from transformers import GenerationConfig

from models import Naive_LLM, Quant_LLM, LLM_Flash, LLM_MultiStream
from datasets import Spider, DialogSum, E2ENLG, create_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="naive")
    parser.add_argument('--sample_size', type=int, default='32')
    args = parser.parse_args()
    
    
    batch_size = 1

    spider = Spider(download=True, train=False)
    spider_prompts = [sample.question for sample in spider]

    dialogsum = DialogSum(download=True, train=False)
    dialogsum_prompts = [sample.summary for sample in dialogsum] # not actually the prompt, need to come up with one

    e2e_nlg = E2ENLG(download=True, train=True)
    e2e_nlg_prompts = [sample.ref for sample in e2e_nlg] # same, not actually the prompt

    print('GPT-Neo 2.7b:')
    if args.model_type == 'naive':
        neo = Naive_LLM("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'quant':
        neo = Quant_LLM("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'flash':
        neo = LLM_Flash("EleutherAI/gpt-neo-2.7B")
    elif args.model_type == 'spec_stream':
        neo = LLM_MultiStream("EleutherAI/gpt-neo-2.7B")
    else:
        assert False, "Model Type not defined"
        
    neo.transformers.eval()
    config = GenerationConfig(max_new_tokens=256, pad_token_id=neo.generate_tokenizer.pad_token_id)

    total_tokens, total_time = 0, 0.0
    # for idx in tqdm(range(0, len(spider_prompts), batch_size)):
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
        print(f"TOKENS : {tokens}, TIME: {elapsed}")
    
    print(f"Spider: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(dialogsum_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"DialogSum: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, args.sample_size)):
        _, tokens, elapsed = neo.generate(e2e_nlg_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"E2E-NLG: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")
