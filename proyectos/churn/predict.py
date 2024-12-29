# Importaciones estándar de Python
import logging

# Bibliotecas de terceros
import pandas as pd
from flask import Flask, request, jsonify

# Importaciones locales
from joblib import load

from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(customer):
    # Preparamos los datos...
    try:
        customer_df = pd.DataFrame([customer])
        customer_df.columns = customer_df.columns.str.lower().str.replace(' ', '_')
        if 'customerid' in customer_df.columns:
                customer_df = customer_df.drop('customerid', axis=1)
    
        numeric_columns = ['tenure', 'monthlycharges', 'totalcharges']
        for col in numeric_columns:
                customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
        
        return customer_df
    
    except Exception as e:
        logging.error(f"Error al preparar los datos: {str(e)}")

def predict_single(customer, model):
    # Hacemos la predicción
    try:
        customer_df = prepare_data(customer)    
        y_pred = model.predict_proba(customer_df)[:, 1]
        return y_pred[0]
    except Exception as e:
        logging.error(f"Error al hacer la predicción: {str(e)}")

try:
    modelo = load('modelo_churn.joblib')
    logger.info("Modelo cargado!")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        customer = request.get_json()
        
        if not customer:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        prediction = predict_single(customer, modelo)
        
        result = {
            'churn_probability': float(prediction),
            'churn': bool(prediction >= 0.5),
            'churn_percent': f"{float(prediction)*100:.2f}%"
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=9696)




