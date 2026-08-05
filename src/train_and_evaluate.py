"""
Real Estate ML Pipeline: 5-Fold Cross Validation & Model Evaluation Benchmark
-----------------------------------------------------------------------------
Evaluates Linear Regression, Random Forest, and Monotonic XGBoost Regressors
across 23 comprehensive property features.
Metrics: R² Score, RMSE ($), MAE ($)
"""

import sys
from pathlib import Path

# Sanitize sys.path to prevent google-cloud-sdk importlib collision & add project root
sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data_processing import load_and_preprocess_data

MONOTONE_CONSTRAINTS = "(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"

def run_ds_benchmark():
    print("=" * 75)
    print("      AMES REAL ESTATE VALUATION MODEL BENCHMARK & EVALUATION      ")
    print("=" * 75)
    
    # 1. Load Preprocessed Data
    X, y = load_and_preprocess_data()
    print(f"\n[1] Dataset Loaded: {X.shape[0]} observations, {X.shape[1]} features.")
    print(f"    Target Variable: SalePrice (Mean: ${y.mean():,.2f}, Median: ${y.median():,.2f}, Std: ${y.std():,.2f})")
    
    # 2. 5-Fold Cross Validation Setup
    print("\n[2] Running 5-Fold Cross-Validation Model Benchmark...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42),
        "Monotonic XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            random_state=42
        )
    }
    
    results = []
    for name, model in models.items():
        cv_res = cross_validate(
            model, X, y, cv=kf,
            scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
        )
        mean_r2 = cv_res['test_r2'].mean()
        mean_rmse = -cv_res['test_rmse'].mean()
        mean_mae = -cv_res['test_mae'].mean()
        results.append({
            "Model Paradigm": name,
            "R^2 Score": f"{mean_r2:.4f}",
            "RMSE ($)": f"${mean_rmse:,.2f}",
            "MAE ($)": f"${mean_mae:,.2f}"
        })
    
    df_results = pd.DataFrame(results)
    print("\n--- MODEL BENCHMARK EVALUATION RESULTS ---")
    print(df_results.to_string(index=False))
    print("------------------------------------------\n")
    
    # 3. Monotonic XGBoost Feature Importances
    prod_model = models["Monotonic XGBoost Regressor"]
    prod_model.fit(X, y)
    importances = prod_model.feature_importances_
    
    print("[3] Monotonic XGBoost Feature Weight Distribution (Gain):")
    feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp:
        print(f"    - {feat:25s}: {imp*100:6.2f}%")
        
    print("\n" + "=" * 75)
    print("             REAL ESTATE BENCHMARK EVALUATION COMPLETE             ")
    print("=" * 75)
    return df_results

if __name__ == '__main__':
    run_ds_benchmark()
