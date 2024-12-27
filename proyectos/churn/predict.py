from joblib import load
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

def predict_single(customer, model):
    customer_df = pd.DataFrame([customer])  
    y_pred = model.predict_proba(customer_df)[:, 1]
    return y_pred[0]

modelo = load('modelo_churn.joblib')

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    prediction = predict_single(customer, modelo)
    churn = prediction >= 0.5
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=9696)




