from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Real Estate ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"
DATASET_PATH = Path(__file__).parent / "AmesHousing.csv"

model = joblib.load(MODEL_PATH)

# Load dataset for market analytics
df_raw = pd.read_csv(DATASET_PATH)
df_clean = df_raw[['Gr Liv Area', 'Bedroom AbvGr', 'Year Built', 'Full Bath', 'SalePrice']].dropna()
df_clean['Price_Per_SqFt'] = df_clean['SalePrice'] / df_clean['Gr Liv Area']

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
    price = float(prediction[0])
    price_per_sqft = price / max(features.Gr_Liv_Area, 1)
    
    # Calculate market percentile comparison
    percentile = float((df_clean['SalePrice'] < price).mean() * 100)
    
    return {
        "predicted_price": price,
        "price_per_sqft": price_per_sqft,
        "market_percentile": round(percentile, 1)
    }

@app.get("/analytics/kpis")
def get_kpis():
    avg_price = float(df_clean['SalePrice'].mean())
    median_price = float(df_clean['SalePrice'].median())
    avg_price_sqft = float(df_clean['Price_Per_SqFt'].mean())
    total_records = int(len(df_clean))
    
    return {
        "avg_price": round(avg_price, 2),
        "median_price": round(median_price, 2),
        "avg_price_sqft": round(avg_price_sqft, 2),
        "total_records": total_records
    }

@app.get("/analytics/trends")
def get_decade_trends():
    df_clean_copy = df_clean.copy()
    df_clean_copy['Decade'] = (df_clean_copy['Year Built'] // 10 * 10).astype(str) + "s"
    trends = df_clean_copy.groupby('Decade').agg(
        avg_price=('SalePrice', 'mean'),
        avg_sqft=('Gr Liv Area', 'mean'),
        sample_count=('SalePrice', 'count')
    ).reset_index()
    
    # Sort chronologically
    trends = trends.sort_values('Decade')
    
    result = []
    for _, row in trends.iterrows():
        result.append({
            "decade": row['Decade'],
            "avg_price": round(float(row['avg_price']), 2),
            "avg_sqft": round(float(row['avg_sqft']), 2),
            "sample_count": int(row['sample_count'])
        })
    return result

@app.get("/analytics/price-vs-sqft")
def get_size_brackets():
    bins = [0, 1000, 1500, 2000, 2500, 10000]
    labels = ["< 1,000", "1,000-1,500", "1,500-2,000", "2,000-2,500", "> 2,500"]
    
    df_clean_copy = df_clean.copy()
    df_clean_copy['Size_Tier'] = pd.cut(df_clean_copy['Gr Liv Area'], bins=bins, labels=labels)
    
    grouped = df_clean_copy.groupby('Size_Tier', observed=False).agg(
        avg_price=('SalePrice', 'mean'),
        avg_price_sqft=('Price_Per_SqFt', 'mean'),
        count=('SalePrice', 'count')
    ).reset_index()
    
    result = []
    for _, row in grouped.iterrows():
        result.append({
            "size_tier": str(row['Size_Tier']),
            "avg_price": round(float(row['avg_price']), 2),
            "avg_price_sqft": round(float(row['avg_price_sqft']), 2),
            "count": int(row['count'])
        })
    return result

@app.get("/analytics/feature-importance")
def get_feature_importance():
    feature_names = ['Living Area (sq ft)', 'Year Built', 'Full Bathrooms', 'Bedrooms']
    importances = model.feature_importances_
    
    # Pair and sort
    items = []
    raw_names = ['Gr Liv Area', 'Year Built', 'Full Bath', 'Bedroom AbvGr']
    for raw, pretty, weight in zip(raw_names, feature_names, importances):
        items.append({
            "feature": pretty,
            "importance": round(float(weight) * 100, 1)
        })
    
    items.sort(key=lambda x: x['importance'], reverse=True)
    return items