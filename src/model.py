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

from src.data_processing import load_and_preprocess_data

MODEL_SAVE_PATH = Path(__file__).parent / "xgb_model.pkl"

def train_and_serialize_model() -> Path:
    """Fits Monotonic XGBoost on preprocessed dataset and serializes to disk."""
    print("=" * 60)
    print("TRAINING & SERIALIZING MONOTONIC XGBOOST MODEL")
    print("=" * 60)
    
    X, y = load_and_preprocess_data()
    
    model = xgb.XGBRegressor(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints='(1, 1, 1, 1, 0, 1, 1)',
        random_state=42
    )
    
    model.fit(X, y)
    
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model trained and serialized to: '{MODEL_SAVE_PATH.resolve()}'")
    
    # Test Sample Predict
    sample_input = pd.DataFrame([{
        'Overall Qual': 8,
        'Gr Liv Area': 2100,
        'Garage Cars': 2,
        'Full Bath': 2,
        'Bedroom AbvGr': 3,
        'Year Built': 2005,
        'Qual_Area_Interaction': 8 * 2100
    }])
    
    pred_val = model.predict(sample_input)[0]
    print(f" -> Sample Property Valuation Estimate: ${pred_val:,.2f}")
    return MODEL_SAVE_PATH

if __name__ == '__main__':
    train_and_serialize_model()
