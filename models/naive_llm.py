from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

import copy

class Naive_LLM(nn.Module):
    def __init__(self, model_name, model_kwargs={}, tok_kwargs={}, device='cuda'):
        super(Naive_LLM, self).__init__()
        self.device = device
        # Load Model
        self.transformers = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.transformers.to(self.device)
        # Load Tokenizer (For Forward)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load Tokenizer (For Generate, It should be in left side)
        self.generate_tokenizer = copy.deepcopy(self.tokenizer)
        self.generate_tokenizer.padding_side="left"
        
    def forward(self, input_texts):
        """"
        Input_text -> Array of texts where length is batch size
        Labels -> Array of texts where length is batch size
        """
        tokenizer_outputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        input_ids = tokenizer_outputs.input_ids
        attention_masks = tokenizer_outputs.attention_mask
        loss = self.transformers(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids).loss
        return loss
    
    def generate(self, input_texts, return_full_text=True, generate_kwargs={}):
        tokenizer_outputs = self.tokenizer(input_texts, padding=True, return_tensors='pt').to(self.device)
        generate_ids = self.transformers.generate(**tokenizer_outputs, **generate_kwargs)
        generated = self.generate_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        
        if return_full_text:
            return generated[0] if isinstance(input_texts, str) else generated
        # If return_full_text is false we should exclude the input prompt
        if isinstance(input_texts, str):
            prompt_length = len(input_texts)
            generated = generated[0][prompt_length:]
        else:
            prompt_lengths = [len(i) for i in input_texts]
            generated = [gen[length:] for (length, gen) in zip(prompt_lengths, generated)]
        return generated