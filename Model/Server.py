from flask import Flask, jsonify, request
import os, sys
sys.path.insert(0, os.getcwd())
from utils import GPUs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


GPUs.init_gpu()
app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4")
model = AutoModelForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4", device_map="auto", torch_dtype=torch.float16).to("cuda")


@app.route('/chat', methods=['POST'])
def chat():
    message = '<s>' + request.form['prompt'] + "</s></s>"
    max_new_tokens = int(request.form.get("max_new_tokens", 256))
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.85, temperature=0.35,
                            repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(tokens[0])
    return {"inputs": message, "reply": res}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=19888)
