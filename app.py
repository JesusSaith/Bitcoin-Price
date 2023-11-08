from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
import fastapi

app = fastapi.FastAPI(title='Bitcoin Price Predictor')

model = load(pathlib.Path('model/bitcoin-v4.joblib'))

class BitcoinPricePredictionData(BaseModel):
    open_price: float = 10000
    close_price: float = 10500
    volume: float = 1000000

class BitcoinPricePredictionOutput(BaseModel):
    price_prediction: float

@app.post('/predict_price', response_model = BitcoinPricePredictionOutput)
def predict_price(data: BitcoinPricePredictionData):
    model_input = np.array([data.open_price, data.close_price, data.volume]).reshape(1,-1)
    result = model.predict(model_input)[0]

    return {'price_prediction': result}
