import re
import json
import os
import requests
from flask import Flask, render_template, request, Blueprint

# from utils import GPUs
# from Model import OpenAssistant

# import Web.chat_demo

# GPUs.init_gpu()
basedir = os.path.abspath(os.path.dirname(__file__))
#app = Blueprint("chat", __name__)
app = Flask(__name__)


# app.register_blueprint(Web.chat_demo.app)


@app.route('/')
def main():
    return render_template('new.html')


@app.route('/feature')
def feature():
    return render_template('feature.html')


@app.route('/block')
def block():
    return render_template('block.html')


@app.route('/chatblock')
def chatblock():
    return render_template('chatblock.html')


@app.route('/pictureblock')
def pictureblock():
    return render_template('pictureblock.html')


@app.route('/main')
def online():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def picture():
    img = request.files.get('file')
    path = basedir + "/static/img/"
    img_name = img.filename
    file_path = path + img_name
    img.save(file_path)
    url = '/static/img/' + img_name
    return url


@app.route('/chat', methods=['POST'])
def chat():
    inputs = request.form['dialog']
    dialog = OpenAssistant.forward(inputs, max_new_tokens=150)
    #dialog = "1111"
    end = "<|endoftext|>" in dialog[len(inputs):]
    reply = dialog[len(inputs):].replace("<|endoftext|>", "")
    return {"reply": reply, "dialog": dialog, "ts": request.form['ts'], 'end': end}

#@app.route("/chat_demo")
#def chat_demo():
#    return render_template("chat_demo.html")


#@app.route("/chat_api", methods=['POST'])
#def chat_api():
#    message = request.form['message']
#    res = requests.post("http://210.28.134.55:19888/chat", {"prompt": message}).json()
#    return res