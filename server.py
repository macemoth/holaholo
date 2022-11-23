from flask import Flask, jsonify, request
import pandas as pd
from io import StringIO
import classifier

server = Flask(__name__)


@server.route('/classify', methods=["POST"])
def classify_request():
    data = request.get_data(as_text=True)
    print(data)
    df = pd.read_csv(StringIO(data))
    c = classifier.classify(df)
    df.iloc[-1,]
    date = str(pd.to_datetime(df["eyeDataTimestamp"].iloc[-1], unit="ms"))
    if len(c) > 0:
        return jsonify([[c[0]][0], date])
    else:
        return "Unknown"
