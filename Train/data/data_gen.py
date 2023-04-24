import os
import tqdm
import json

process_ids = [25, 26, 27, 28]
os.environ['HF_DATASETS_CACHE'] = "cache"
os.environ['HUGGINGFACE_HUB_CACHE '] = "cache"
os.environ['TRANSFORMERS_CACHE'] = "cache"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from Train.data import Zhihu


tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/oasst-sft-4-pythia-12b-epoch-3.5", cache_dir='cache')
model = GPTNeoXForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/oasst-sft-4-pythia-12b-epoch-3.5", device_map="auto", torch_dtype=torch.float16, cache_dir='cache')
model.eval()

dataset = Zhihu.build_dataset()
max_tokens = 1024
data_gen = []

for pid in process_ids:
    with torch.no_grad():
        for t in tqdm.tqdm(sorted(set(dataset['title']))[pid*500: (pid+1)*500], desc=f"{pid}"):
            message = f"<|prompter|>{t}<|endoftext|><|assistant|>"
            inputs = tokenizer(message, return_tensors="pt").to(model.device)
            tokens = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.8)
            res = tokenizer.decode(tokens[0])
            data_gen.append((t, res))

    with open(f"data_oasst_part{pid}.json", "w") as f:
        json.dump(data_gen, f, indent=2, ensure_ascii=False)
