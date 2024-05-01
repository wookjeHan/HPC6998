import gc
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from models import LLM_MultiStream
import huggingface.transformers.models.gpt_neo.modeling_gpt_neo as gpt_neo
from models.multi_stream_attention import NgramMultiheadAttention
from datasets import Spider, create_dataloader


if __name__ == "__main__":
    batch_size = 1

    spider = Spider(download=True, train=False)
    spider_prompts = [sample.question for sample in spider]

    neo = LLM_MultiStream("EleutherAI/gpt-neo-2.7B")
    neo.transformers.eval()
    config = GenerationConfig(max_new_tokens=256, pad_token_id=neo.generate_tokenizer.pad_token_id)

    total_tokens, total_time = 0, 0.0
    for idx in tqdm(range(0, 10, batch_size)):
        _, tokens, elapsed = neo.generate(spider_prompts[idx:idx+batch_size], generate_kwargs={"generation_config": config})
        total_tokens += tokens
        total_time += elapsed
    print(f"Spider: Tokens/sec: {total_tokens / total_time} ({total_time} seconds)")


