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
    content = request.form['content']
    reply = OpenAssistant.forward(content)
    reply = re.search(r"<\|assistant\|>(.*)(<\|endoftext\|>)?", reply).group(1)
    return reply
