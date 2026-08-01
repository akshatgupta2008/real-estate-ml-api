from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Real Estate ML API")

# Allow web browsers to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load('xgb_model.pkl')

class PropertyFeatures(BaseModel):
    Gr_Liv_Area: float
    Bedroom_AbvGr: int
    Year_Built: int
    Full_Bath: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate ML API!"}

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    input_data = pd.DataFrame([{
        'Gr Liv Area': features.Gr_Liv_Area,
        'Bedroom AbvGr': features.Bedroom_AbvGr,
        'Year Built': features.Year_Built,
        'Full Bath': features.Full_Bath
    }])
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0])}