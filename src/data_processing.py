"""
Real Estate ML Pipeline: Comprehensive Data Processing & Feature Engineering
-----------------------------------------------------------------------------
Ingests and cleans the Ames Housing dataset, encoding structural, quality,
area, room, and temporal metrics into a rich feature matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "AmesHousing.csv"
TARGET_COL = 'SalePrice'

QUAL_MAP = {
    'Very_Poor': 1, 'Poor': 2, 'Fair': 3, 'Below_Average': 4,
    'Average': 5, 'Above_Average': 6, 'Good': 7, 'Very_Good': 8,
    'Excellent': 9, 'Very_Excellent': 10
}

RATING_MAP = {
    'Po': 1, 'Poor': 1,
    'Fa': 2, 'Fair': 2,
    'TA': 3, 'Typical': 3, 'Average': 3,
    'Gd': 4, 'Good': 4,
    'Ex': 5, 'Excellent': 5
}

NUMERICAL_COLS = [
    'Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF', 'First_Flr_SF',
    'Second_Flr_SF', 'Garage_Area', 'Garage Cars', 'Full Bath',
    'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr', 'TotRms_AbvGrd',
    'Fireplaces', 'Lot_Area', 'Wood_Deck_SF', 'Open_Porch_SF',
    'Year Built', 'Year_Remod_Add'
]

FEATURE_COLUMNS = [
    'Overall Qual', 'Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF',
    'First_Flr_SF', 'Second_Flr_SF', 'Garage_Area', 'Garage Cars',
    'Full Bath', 'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr',
    'TotRms_AbvGrd', 'Fireplaces', 'Lot_Area', 'Wood_Deck_SF',
    'Open_Porch_SF', 'Year Built', 'Year_Remod_Add', 'Exter_Qual',
    'Kitchen_Qual', 'Qual_Area_Interaction', 'Total_Living_SF'
]

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
    
    # Anomaly Detection & Outlier Removal (De Cock, 2011 recommendation: Gr Liv Area < 4000 sq ft)
    df_clean = df_raw[(df_raw['Gr Liv Area'] < 4000) & (df_raw[TARGET_COL] > 0)].copy()
    
    # Process Overall Qual (support numeric or categorical string)
    if df_clean['Overall Qual'].dtype == object or isinstance(df_clean['Overall Qual'].dtype, pd.CategoricalDtype):
        df_clean['Overall Qual'] = df_clean['Overall Qual'].map(QUAL_MAP).fillna(5)
    df_clean['Overall Qual'] = pd.to_numeric(df_clean['Overall Qual'], errors='coerce').fillna(5).astype(float)
    
    # Process Ordinal Quality Metrics
    for q_col in ['Exter_Qual', 'Kitchen_Qual']:
        if q_col in df_clean.columns:
            if df_clean[q_col].dtype == object:
                df_clean[q_col] = df_clean[q_col].map(RATING_MAP).fillna(3)
            df_clean[q_col] = pd.to_numeric(df_clean[q_col], errors='coerce').fillna(3).astype(float)
            
    # Convert numerical features & impute NaNs cleanly
    for col in NUMERICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(float)
            
    # Feature Engineering
    df_clean['Qual_Area_Interaction'] = df_clean['Overall Qual'] * df_clean['Gr Liv Area']
    df_clean['Total_Living_SF'] = df_clean['Gr Liv Area'] + df_clean['Total_Bsmt_SF']
    
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COL].astype(float)
    
    return X, y

def preprocess_input_dict(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Sanitizes and transforms a single input dictionary/payload into a DataFrame
    ready for model inference.
    """
    raw = input_dict.copy()
    
    # Parse Overall Qual
    qual_val = raw.get('Overall Qual', raw.get('Overall_Qual', 5))
    if isinstance(qual_val, str):
        qual_val = QUAL_MAP.get(qual_val, 5)
    qual_val = float(qual_val)
    
    # Parse ratings
    exter_val = raw.get('Exter_Qual', 3)
    if isinstance(exter_val, str):
        exter_val = RATING_MAP.get(exter_val, 3)
        
    kitchen_val = raw.get('Kitchen_Qual', 3)
    if isinstance(kitchen_val, str):
        kitchen_val = RATING_MAP.get(kitchen_val, 3)
        
    gr_liv = float(raw.get('Gr Liv Area', raw.get('Gr_Liv_Area', 1500)))
    bsmt_sf = float(raw.get('Total_Bsmt_SF', raw.get('Total Bsmt SF', 800)))
    
    record = {
        'Overall Qual': qual_val,
        'Overall_Cond': float(raw.get('Overall_Cond', raw.get('Overall Cond', 5))),
        'Gr Liv Area': gr_liv,
        'Total_Bsmt_SF': bsmt_sf,
        'First_Flr_SF': float(raw.get('First_Flr_SF', raw.get('1st Flr SF', gr_liv * 0.6))),
        'Second_Flr_SF': float(raw.get('Second_Flr_SF', raw.get('2nd Flr SF', gr_liv * 0.4))),
        'Garage_Area': float(raw.get('Garage_Area', raw.get('Garage Area', 400))),
        'Garage Cars': float(raw.get('Garage Cars', raw.get('Garage_Cars', 2))),
        'Full Bath': float(raw.get('Full Bath', raw.get('Full_Bath', 2))),
        'Half_Bath': float(raw.get('Half_Bath', raw.get('Half Bath', 0))),
        'Bsmt_Full_Bath': float(raw.get('Bsmt_Full_Bath', raw.get('Bsmt Full Bath', 0))),
        'Bedroom AbvGr': float(raw.get('Bedroom AbvGr', raw.get('Bedroom_AbvGr', 3))),
        'TotRms_AbvGrd': float(raw.get('TotRms_AbvGrd', raw.get('TotRms AbvGrd', 6))),
        'Fireplaces': float(raw.get('Fireplaces', 1)),
        'Lot_Area': float(raw.get('Lot_Area', raw.get('Lot Area', 10000))),
        'Wood_Deck_SF': float(raw.get('Wood_Deck_SF', raw.get('Wood Deck SF', 0))),
        'Open_Porch_SF': float(raw.get('Open_Porch_SF', raw.get('Open Porch SF', 0))),
        'Year Built': float(raw.get('Year Built', raw.get('Year_Built', 2000))),
        'Year_Remod_Add': float(raw.get('Year_Remod_Add', raw.get('Year Remod Add', 2005))),
        'Exter_Qual': float(exter_val),
        'Kitchen_Qual': float(kitchen_val),
        'Qual_Area_Interaction': qual_val * gr_liv,
        'Total_Living_SF': gr_liv + bsmt_sf,
    }
    
    return pd.DataFrame([record])[FEATURE_COLUMNS]

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print("[SUCCESS] Data Processing Module Verified!")
    print(f" -> Cleaned Dataset Observations: {X.shape[0]}")
    print(f" -> Feature Count: {X.shape[1]} | Features: {list(X.columns)}")
    print(f" -> Target Mean SalePrice: ${y.mean():,.2f}")
