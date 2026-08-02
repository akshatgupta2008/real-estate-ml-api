from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
feature_cols = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Full Bath', 'Bedroom AbvGr', 'Year Built']
df_clean = df_raw[feature_cols + ['Neighborhood', 'SalePrice']].dropna().copy()
df_clean['Price_Per_SqFt'] = df_clean['SalePrice'] / df_clean['Gr Liv Area']

class PropertyFeatures(BaseModel):
    Gr_Liv_Area: float = Field(..., gt=0)
    Bedroom_AbvGr: int = Field(..., ge=0)
    Year_Built: int = Field(..., ge=1800)
    Full_Bath: int = Field(..., ge=0)
    Overall_Qual: int = Field(6, ge=1, le=10)
    Garage_Cars: int = Field(1, ge=0, le=5)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate ML API!"}

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    input_data = pd.DataFrame([{
        'Overall Qual': features.Overall_Qual,
        'Gr Liv Area': features.Gr_Liv_Area,
        'Garage Cars': features.Garage_Cars,
        'Full Bath': features.Full_Bath,
        'Bedroom AbvGr': features.Bedroom_AbvGr,
        'Year Built': features.Year_Built
    }])
    
    prediction = model.predict(input_data)
    price = float(prediction[0])
    price_per_sqft = price / max(features.Gr_Liv_Area, 1)
    
    # Calculate market percentile rank
    percentile = float((df_clean['SalePrice'] < price).mean() * 100)
    
    return {
        "predicted_price": price,
        "price_per_sqft": price_per_sqft,
        "market_percentile": round(percentile, 1)
    }

@app.post("/analytics/roi-simulator")
def calculate_roi(features: PropertyFeatures):
    base_df = pd.DataFrame([{
        'Overall Qual': features.Overall_Qual,
        'Gr Liv Area': features.Gr_Liv_Area,
        'Garage Cars': features.Garage_Cars,
        'Full Bath': features.Full_Bath,
        'Bedroom AbvGr': features.Bedroom_AbvGr,
        'Year Built': features.Year_Built
    }])
    base_price = float(model.predict(base_df)[0])
    
    # Upgrade 1: +1 Full Bathroom
    df_bath = base_df.copy()
    df_bath['Full Bath'] += 1
    price_bath = float(model.predict(df_bath)[0])
    
    # Upgrade 2: +1 Quality Tier (max 10)
    df_qual = base_df.copy()
    df_qual['Overall Qual'] = min(features.Overall_Qual + 1, 10)
    price_qual = float(model.predict(df_qual)[0])
    
    # Upgrade 3: +1 Garage Space (max 4)
    df_gar = base_df.copy()
    df_gar['Garage Cars'] = min(features.Garage_Cars + 1, 4)
    price_gar = float(model.predict(df_gar)[0])
    
    # Upgrade 4: +500 sq ft Living Area
    df_sqft = base_df.copy()
    df_sqft['Gr Liv Area'] += 500
    price_sqft = float(model.predict(df_sqft)[0])
    
    return {
        "base_price": base_price,
        "upgrades": [
            {
                "name": "Add 1 Full Bathroom",
                "added_value": round(max(0, price_bath - base_price), 0),
                "new_price": round(price_bath, 0)
            },
            {
                "name": "Upgrade Quality (+1 Level)",
                "added_value": round(max(0, price_qual - base_price), 0),
                "new_price": round(price_qual, 0)
            },
            {
                "name": "Add 1 Garage Car Space",
                "added_value": round(max(0, price_gar - base_price), 0),
                "new_price": round(price_gar, 0)
            },
            {
                "name": "Add 500 sq ft Living Space",
                "added_value": round(max(0, price_sqft - base_price), 0),
                "new_price": round(price_sqft, 0)
            }
        ]
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

@app.get("/analytics/neighborhoods")
def get_neighborhood_analytics():
    grouped = df_clean.groupby('Neighborhood').agg(
        avg_price=('SalePrice', 'mean'),
        avg_price_sqft=('Price_Per_SqFt', 'mean'),
        count=('SalePrice', 'count')
    ).reset_index()
    
    # Filter neighborhoods with at least 25 properties and sort by avg price descending
    grouped = grouped[grouped['count'] >= 20].sort_values('avg_price', ascending=False)
    
    result = []
    for _, row in grouped.iterrows():
        result.append({
            "neighborhood": row['Neighborhood'],
            "avg_price": round(float(row['avg_price']), 0),
            "avg_price_sqft": round(float(row['avg_price_sqft']), 1),
            "count": int(row['count'])
        })
    return result

@app.get("/analytics/trends")
def get_decade_trends():
    df_clean_copy = df_clean.copy()
    df_clean_copy['Decade'] = (df_clean_copy['Year Built'] // 10 * 10).astype(str) + "s"
    trends = df_clean_copy.groupby('Decade').agg(
        avg_price=('SalePrice', 'mean'),
        avg_sqft=('Gr Liv Area', 'mean'),
        sample_count=('SalePrice', 'count')
    ).reset_index()
    
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
    feature_names = ['Overall Quality Rating', 'Living Area (sq ft)', 'Garage Capacity (Cars)', 'Full Bathrooms', 'Year Built', 'Bedrooms']
    importances = model.feature_importances_
    
    raw_names = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Full Bath', 'Year Built', 'Bedroom AbvGr']
    items = []
    for raw, pretty, weight in zip(raw_names, feature_names, importances):
        items.append({
            "feature": pretty,
            "importance": round(float(weight) * 100, 1)
        })
    
    items.sort(key=lambda x: x['importance'], reverse=True)
    return items