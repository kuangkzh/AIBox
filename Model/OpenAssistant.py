import os
import gc
import torch
import warnings
from transformers import AutoTokenizer, GPTNeoXForCausalLM


tokenizer, model = None, None


def forward(input_text, max_new_tokens=200):
    global tokenizer, model

    if not tokenizer:
        if not os.path.exists("/home/nlper_data/kuangzh/oasst-sft-4-pythia-12b-epoch-3.5"):
            warnings.warn("The OpenAssistant model is not found, skipping...")
            return ""

        tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/oasst-sft-4-pythia-12b-epoch-3.5", cache_dir='cache')
        model = GPTNeoXForCausalLM.from_pretrained("/home/nlper_data/kuangzh/oasst-sft-4-pythia-12b-epoch-3.5", device_map="auto", cache_dir='cache')

    # message = f"<|prompter|>{input_text}<|endoftext|><|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    res = tokenizer.decode(tokens[0])
    return res


def empty_cache():
    global tokenizer, model
    tokenizer, model = None, None
    gc.collect()
    torch.cuda.empty_cache()
