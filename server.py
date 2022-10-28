from flask import Flask, jsonify, request
import pandas as pd
from io import StringIO
import classifier

server = Flask(__name__)
server.config.update(SERVER_NAME='127.0.0.1:5000')


@server.route('/classify', methods=["POST"])
def classify_request():
    data = request.get_data(as_text=True)
    df = pd.read_csv(StringIO(data))
    c = classifier.classify(df)
    if len(c) > 0:
        return[c[0]]
    else:
        return "Unknown"
