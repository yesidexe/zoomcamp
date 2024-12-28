from joblib import load
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

def prepare_data(customer):
    customer_df = pd.DataFrame([customer])
    customer_df.columns = customer_df.columns.str.lower().str.replace(' ', '_')
    if 'customerid' in customer_df.columns:
            customer_df = customer_df.drop('customerid', axis=1)
    
    numeric_columns = ['tenure', 'monthlycharges', 'totalcharges']
    for col in numeric_columns:
            customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
    
    null_columns = customer_df.columns[customer_df.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Missing values found in columns: {null_columns}")
    
    return customer_df

def predict_single(customer, model):
    customer_df = prepare_data(customer)    
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
            'churn': bool(churn),
            'churn_percent': f"{float(prediction)*100:.2f}%"
        }
        return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=9696)




