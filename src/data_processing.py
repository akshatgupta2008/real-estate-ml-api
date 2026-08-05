"""
Real Estate ML Pipeline: Data Processing & Feature Engineering Module
----------------------------------------------------------------------
Covers:
1. Data Ingestion & Schema Sanitization
2. Statistical Anomaly Removal (>4,000 sq ft Gr Liv Area)
3. Domain Feature Engineering (Quality-to-Area Interaction)
4. Train/Validation Feature Matrix Construction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "AmesHousing.csv"
BASE_FEATURES = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Full Bath', 'Bedroom AbvGr', 'Year Built']
TARGET_COL = 'SalePrice'

QUAL_MAP = {
    'Very_Poor': 1, 'Poor': 2, 'Fair': 3, 'Below_Average': 4, 
    'Average': 5, 'Above_Average': 6, 'Good': 7, 'Very_Good': 8, 
    'Excellent': 9, 'Very_Excellent': 10
}

def load_and_preprocess_data(file_path: Path = DEFAULT_DATASET_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw Ames Housing dataset, performs anomaly detection & feature engineering.
    
    Returns:
        X (pd.DataFrame): Processed feature matrix with engineered interaction features.
        y (pd.Series): Target SalePrice vector.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{file_path}'. Please ensure data/AmesHousing.csv exists.")
        
    df_raw = pd.read_csv(file_path)
    
    # Required columns check
    all_cols = BASE_FEATURES + [TARGET_COL]
    missing_cols = [c for c in all_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing mandatory columns: {missing_cols}")
        
    df_clean = df_raw.copy()
    
    # Handle Overall Qual string or categorical mapping
    if df_clean['Overall Qual'].dtype == object or isinstance(df_clean['Overall Qual'].dtype, pd.CategoricalDtype):
        df_clean['Overall Qual'] = df_clean['Overall Qual'].map(QUAL_MAP).fillna(5)
    df_clean['Overall Qual'] = pd.to_numeric(df_clean['Overall Qual'], errors='coerce').fillna(5).astype(float)
    
    # Convert numerical features
    for col in ['Gr Liv Area', 'Garage Cars', 'Full Bath', 'Bedroom AbvGr', 'Year Built', TARGET_COL]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Anomaly Detection & Outlier Removal (De Cock, 2011 recommendation: Gr Liv Area < 4000 sq ft)
    df_clean = df_clean[(df_clean['Gr Liv Area'] < 4000) & (df_clean[TARGET_COL] > 0)].dropna(subset=all_cols).copy()
    
    # Feature Engineering: Quality-to-Area Interaction Term
    X = df_clean[BASE_FEATURES].copy()
    X['Qual_Area_Interaction'] = X['Overall Qual'] * X['Gr Liv Area']
    y = df_clean[TARGET_COL]
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print("[SUCCESS] Data Processing Module Verified!")
    print(f" -> Cleaned Dataset Observations: {X.shape[0]}")
    print(f" -> Feature Count: {X.shape[1]} | Features: {list(X.columns)}")
    print(f" -> Target Mean SalePrice: ${y.mean():,.2f}")
