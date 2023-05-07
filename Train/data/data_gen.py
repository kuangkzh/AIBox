import os
import tqdm
import json

process_ids = [5]
print(process_ids)
os.environ['HF_DATASETS_CACHE'] = "cache"
os.environ['HUGGINGFACE_HUB_CACHE '] = "cache"
os.environ['TRANSFORMERS_CACHE'] = "cache"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, AutoModelForCausalLM
from Train.data import Zhihu


# tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/oasst-sft-4-pythia-12b-epoch-3.5")
# model = GPTNeoXForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/oasst-sft-4-pythia-12b-epoch-3.5", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4")
model = AutoModelForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4", device_map="auto", torch_dtype=torch.float16).to("cuda")
model.eval()

dataset = Zhihu.build_dataset()
max_tokens = 1024
data_gen = []

for pid in process_ids:
    with torch.no_grad():
        for t in tqdm.tqdm(sorted(set(dataset['title']))[pid*5000: (pid+1)*5000], desc=f"{pid}"):
            # message = f"<|prompter|>{t}<|endoftext|><|assistant|>"
            message = f"<s>{t}</s></s>"
            inputs = tokenizer(message, return_tensors="pt").to(model.device)
            tokens = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.85, temperature=0.35,
                             repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
            res = tokenizer.decode(tokens[0])
            data_gen.append((t, res))

    with open(f"data_firefly_part{pid}.json", "w") as f:
        json.dump(data_gen, f, indent=2, ensure_ascii=False)
