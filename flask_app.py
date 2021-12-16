from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model_rf.pkl')

@app.route('/predict/', methods=['POST'])
def predict():
    json_ = request.json
    query = pd.DataFrame(json_)
    prediction = model.predict(query.values)
    return jsonify(list(prediction))

@app.route('/')
def index():
    return "<h1>La prediction se trouve sur /predict/ </h1>"


