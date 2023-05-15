import requests
from flask import Blueprint, render_template, request


app = Blueprint("chat", __name__)


@app.route("/chat_demo")
def chat_demo():
    return render_template("chat_demo.html")


@app.route("/chat_api", methods=['POST'])
def chat_api():
    message = request.form['message']
    res = requests.post("http://210.28.134.55:19888/chat", {"prompt": message}).json()
    return res
