from joblib import load
from fastapi import FastAPI, Request
import uvicorn

modelo = load('modelo_churn.joblib')
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    customer_data = await request.json()
    
    customer_data_list = list(customer_data.values())
        
    y_pred = modelo.predict([customer_data_list])
    churn = y_pred >= 0.5
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9696)




