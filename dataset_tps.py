import gc
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from models import Naive_LLM
from datasets.spider import Spider
from datasets.dialogsum import DialogSum
from datasets.e2e_nlg import E2ENLG
from datasets.utils import create_dataloader


if __name__ == "__main__":
    batch_size = 32

    spider = Spider(download=True, train=False)
    spider_prompts = [sample.question for sample in spider]
    # print(spider_prompts[:5])

    dialogsum = DialogSum(download=True, train=False)
    dialogsum_prompts = [sample.summary for sample in dialogsum] # not actually the prompt, need to come up with one
    # print(dialogsum_prompts[:5])

    e2e_nlg = E2ENLG(download=True, train=False)
    e2e_nlg_prompts = [sample.ref for sample in e2e_nlg] # same, not actually the prompt
    # print(e2e_nlg_prompts[:5])

    print('Llama 3b:')
    llama = Naive_LLM("openlm-research/open_llama_3b_v2")
    llama.transformers.eval()
    config = GenerationConfig(max_new_tokens=256)

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(spider_prompts), batch_size)):
        _, tokens, elapsed = llama.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"Spider: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(dialogsum_prompts), batch_size)):
        _, tokens, elapsed = llama.generate(dialogsum_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"DialogSum: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(e2e_nlg_prompts), batch_size)):
        _, tokens, elapsed = llama.generate(e2e_nlg_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"E2E-NLG: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    del llama
    gc.collect()
    torch.cuda.empty_cache()

    print('GPT-Neo 2.7b:')
    neo = Naive_LLM("EleutherAI/gpt-neo-2.7B")
    neo.transformers.eval()
    config = GenerationConfig(max_new_tokens=256, pad_token_id=neo.generate_tokenizer.pad_token_id)

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(spider_prompts), batch_size)):
        _, tokens, elapsed = neo.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"Spider: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(dialogsum_prompts), batch_size)):
        _, tokens, elapsed = neo.generate(dialogsum_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"DialogSum: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, len(e2e_nlg_prompts), batch_size)):
        _, tokens, elapsed = neo.generate(e2e_nlg_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"E2E-NLG: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")

