from joblib import load
import pandas as pd
from fastapi import FastAPI, Request
import uvicorn

modelo = load('modelo_churn.joblib')
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    customer_data = await request.json()
    
    expected_columns = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'tenure', 'phoneservice', 'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod', 'monthlycharges', 'totalcharges'
    ]
    
    customer_data_list = pd.DataFrame([customer_data], columns=expected_columns)
        
    y_pred = modelo.predict(customer_data_list)
    churn = y_pred[0] >= 0.5
    
    result = {
        'churn_probability': float(y_pred[0]),
        'churn': bool(churn)
    }
    
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9696)




