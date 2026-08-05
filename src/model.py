"""
Real Estate ML Pipeline: Model Training & Serialization Module
----------------------------------------------------------------
Trains and serializes production Monotonic XGBoost Regressor model.
"""

import sys
from pathlib import Path

# Sanitize sys.path to prevent google-cloud-sdk importlib collision & add project root
sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from src.data_processing import load_and_preprocess_data, preprocess_input_dict, FEATURE_COLUMNS

MODEL_SAVE_PATH = Path(__file__).parent / "xgb_model.pkl"
MONOTONE_CONSTRAINTS = "(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"

def load_or_train_model():
    """Loads serialized XGBoost model or trains it if missing."""
    if not MODEL_SAVE_PATH.exists():
        train_and_serialize_model()
    return joblib.load(MODEL_SAVE_PATH)


def train_and_serialize_model() -> Path:
    """Fits Monotonic XGBoost on preprocessed dataset and serializes to disk."""
    print("=" * 60)
    print("TRAINING & SERIALIZING MONOTONIC XGBOOST MODEL")
    print("=" * 60)
    
    X, y = load_and_preprocess_data()
    
    model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=42
    )
    
    model.fit(X, y)
    
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model trained and serialized to: '{MODEL_SAVE_PATH.resolve()}'")
    
    # Test Sample Predict
    sample_raw = {
        'Overall Qual': 'Good',
        'Overall_Cond': 6,
        'Gr Liv Area': 2100,
        'Total_Bsmt_SF': 1100,
        'First_Flr_SF': 1200,
        'Second_Flr_SF': 900,
        'Garage_Area': 500,
        'Garage Cars': 2,
        'Full Bath': 2,
        'Half_Bath': 1,
        'Bsmt_Full_Bath': 1,
        'Bedroom AbvGr': 3,
        'TotRms_AbvGrd': 7,
        'Fireplaces': 1,
        'Lot_Area': 10500,
        'Wood_Deck_SF': 140,
        'Open_Porch_SF': 50,
        'Year Built': 2005,
        'Year_Remod_Add': 2008,
        'Exter_Qual': 'Gd',
        'Kitchen_Qual': 'Gd'
    }
    
    sample_df = preprocess_input_dict(sample_raw)
    pred_val = model.predict(sample_df)[0]
    print(f" -> Sample Property Valuation Estimate: ${pred_val:,.2f}")
    return MODEL_SAVE_PATH

if __name__ == '__main__':
    train_and_serialize_model()
