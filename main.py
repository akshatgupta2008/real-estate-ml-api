from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Real Estate ML API")

# Load the saved model trained on the Ames dataset
model = joblib.load('xgb_model.pkl')

# Define the exact inputs the user will send
class PropertyFeatures(BaseModel):
    Gr_Liv_Area: float
    Bedroom_AbvGr: int
    Year_Built: int
    Full_Bath: int

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    # Map the incoming API variables back to the exact column names the model learned
    input_data = pd.DataFrame([{
        'Gr Liv Area': features.Gr_Liv_Area,
        'Bedroom AbvGr': features.Bedroom_AbvGr,
        'Year Built': features.Year_Built,
        'Full Bath': features.Full_Bath
    }])
    
    # Predict
    prediction = model.predict(input_data)
    
    return {"predicted_price": float(prediction[0])}