import re

from flask import Flask, render_template, request
from utils import GPUs
from Model import OpenAssistant


GPUs.init_gpu()

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/chat', methods=['POST'])
def chat():
    inputs = request.form['dialog']
    dialog = OpenAssistant.forward(inputs, max_new_tokens=150)
    end = "<|endoftext|>" in dialog[len(inputs):]
    reply = dialog[len(inputs):].replace("<|endoftext|>", "")
    return {"reply": reply, "dialog": dialog, "ts": request.form['ts'], 'end': end}
