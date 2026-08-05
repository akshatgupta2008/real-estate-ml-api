"""
Real Estate ML Pipeline: Automated Unit & Integration Tests
------------------------------------------------------------
Tests data processing, feature engineering, model prediction, and FastAPI endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from src.data_processing import load_and_preprocess_data, preprocess_input_dict, FEATURE_COLUMNS
from src.model import load_or_train_model, train_and_serialize_model, MODEL_SAVE_PATH
from src.api import app

def get_test_client():
    return TestClient(app)

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_data_loading_and_preprocessing():
    X, y = load_and_preprocess_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] > 2000
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert not X.isnull().values.any()
    assert (y > 0).all()

def test_preprocess_input_dict():
    sample_input = {
        "Gr Liv Area": 2100,
        "Overall Qual": "Good",
        "Year Built": 2010,
        "Garage Cars": 2,
        "Total_Bsmt_SF": 1000,
        "Full Bath": 2
    }
    df_processed = preprocess_input_dict(sample_input)
    assert isinstance(df_processed, pd.DataFrame)
    assert df_processed.shape == (1, len(FEATURE_COLUMNS))
    assert df_processed['Overall Qual'].iloc[0] == 7.0
    assert df_processed['Qual_Area_Interaction'].iloc[0] == 7.0 * 2100.0

def test_model_inference():
    model = load_or_train_model()
    sample_df = preprocess_input_dict({"Gr Liv Area": 2000, "Overall Qual": 8})
    prediction = model.predict(sample_df)[0]
    assert isinstance(prediction, (float, np.floating, np.float32, np.float64))
    assert prediction > 50000.0

def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_api_features_endpoint(client):
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == len(FEATURE_COLUMNS)

def test_api_predict_endpoint(client):
    payload = {
        "Gr Liv Area": 2200.0,
        "Overall Qual": 8,
        "Year Built": 2010,
        "Garage Cars": 2,
        "Total_Bsmt_SF": 1200.0,
        "Full Bath": 2,
        "Exter_Qual": "Gd",
        "Kitchen_Qual": "Gd"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimated_valuation" in data
    assert data["estimated_valuation"] > 100000.0
    assert "formatted_valuation" in data
    assert data["formatted_valuation"].startswith("$")
