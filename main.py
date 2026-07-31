from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Real Estate ML API")

# Load the saved model
model = joblib.load('xgb_model.pkl')

# Tell the API what inputs to expect
class PropertyFeatures(BaseModel):
    sqft: float
    bedrooms: int
    age: int

# Create the prediction link
@app.post("/predict")
def predict_price(features: PropertyFeatures):
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0])}