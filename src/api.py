"""
Real Estate ML Pipeline: FastAPI Serving API & Web GUI Dashboard
----------------------------------------------------------------
Exposes RESTful HTTP endpoints for real-time model predictions
and serves a visual HTML GUI testing dashboard at root (/).
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Sanitize sys.path
sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.data_processing import preprocess_input_dict, FEATURE_COLUMNS
from src.model import MODEL_SAVE_PATH, train_and_serialize_model

app = FastAPI(
    title="Real Estate Monotonic XGBoost Valuation API",
    description="Automated Valuation Model (AVM) API providing single-family home price predictions enforced with domain monotonicity constraints.",
    version="1.0.0"
)

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_SAVE_PATH.exists():
            train_and_serialize_model()
        _model = joblib.load(MODEL_SAVE_PATH)
    return _model

class PropertyPredictionRequest(BaseModel):
    gr_liv_area: float = Field(..., alias="Gr Liv Area", description="Above ground living area in sq ft", example=2100.0)
    overall_qual: Any = Field(..., alias="Overall Qual", description="Overall property build quality rating (1-10)", example=8)
    year_built: int = Field(2005, alias="Year Built", description="Construction year", example=2008)
    garage_cars: int = Field(2, alias="Garage Cars", description="Garage vehicle capacity", example=2)
    total_bsmt_sf: float = Field(1000.0, alias="Total_Bsmt_SF", description="Basement area in sq ft", example=1100.0)
    full_bath: int = Field(2, alias="Full Bath", description="Full bathrooms", example=2)
    half_bath: int = Field(1, alias="Half_Bath", description="Half bathrooms", example=1)
    exter_qual: str = Field("Gd", alias="Exter_Qual", description="Exterior quality ('Ex', 'Gd', 'TA', 'Fa')", example="Gd")
    kitchen_qual: str = Field("Gd", alias="Kitchen_Qual", description="Kitchen quality ('Ex', 'Gd', 'TA', 'Fa')", example="Gd")

    class Config:
        populate_by_name = True

class PropertyPredictionResponse(BaseModel):
    estimated_valuation: float
    formatted_valuation: str
    currency: str = "USD"
    model_version: str = "Monotonic-XGBoost-v1"
    status: str = "success"

@app.get("/", response_class=HTMLResponse)
def read_root_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real Estate ML Valuation Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; padding-bottom: 50px; }
            .navbar { background-color: #1e293b; border-bottom: 1px solid #334155; }
            .card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid #334155; border-radius: 16px; }
            .form-control, .form-select { background-color: #0f172a; border: 1px solid #475569; color: #f8fafc; border-radius: 8px; }
            .form-control:focus, .form-select:focus { background-color: #1e293b; color: #fff; border-color: #10b981; box-shadow: 0 0 0 0.25rem rgba(16, 185, 129, 0.25); }
            .btn-primary { background: linear-gradient(135deg, #10b981 0%, #059669 100%); border: none; font-weight: 700; border-radius: 8px; padding: 12px 24px; }
            .btn-primary:hover { background: linear-gradient(135deg, #059669 0%, #047857 100%); }
            .val-badge { font-size: 2.8rem; font-weight: 800; color: #34d399; letter-spacing: -0.02em; }
            .label-title { font-weight: 600; font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark mb-4 py-3">
            <div class="container">
                <span class="navbar-brand mb-0 h1 fw-bold fs-4">🏡 Real Estate AVM Predictor</span>
                <span class="badge bg-success px-3 py-2 fs-6">Monotonic XGBoost (R² = 89.3%)</span>
            </div>
        </nav>
        
        <div class="container">
            <div class="row g-4">
                <!-- Input Form -->
                <div class="col-lg-7">
                    <div class="card p-4 shadow-lg">
                        <h4 class="fw-bold mb-3 text-white">🛠️ Property Feature Specs</h4>
                        <form id="val-form">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <label class="form-label label-title">Living Area (sq ft)</label>
                                    <input type="number" id="gr_liv_area" class="form-control" value="2100" min="500" max="4000" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Basement Area (sq ft)</label>
                                    <input type="number" id="total_bsmt_sf" class="form-control" value="1100" min="0" max="3000">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Overall Build Quality (1-10)</label>
                                    <input type="number" id="overall_qual" class="form-control" value="8" min="1" max="10" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Year Built</label>
                                    <input type="number" id="year_built" class="form-control" value="2008" min="1870" max="2026" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Garage Capacity (cars)</label>
                                    <input type="number" id="garage_cars" class="form-control" value="2" min="0" max="4">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Full Bathrooms</label>
                                    <input type="number" id="full_bath" class="form-control" value="2" min="1" max="4">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Exterior Quality</label>
                                    <select id="exter_qual" class="form-select">
                                        <option value="Ex">Excellent</option>
                                        <option value="Gd" selected>Good</option>
                                        <option value="TA">Typical</option>
                                        <option value="Fa">Fair</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label label-title">Kitchen Quality</label>
                                    <select id="kitchen_qual" class="form-select">
                                        <option value="Ex">Excellent</option>
                                        <option value="Gd" selected>Good</option>
                                        <option value="TA">Typical</option>
                                        <option value="Fa">Fair</option>
                                    </select>
                                </div>
                                <div class="col-12 mt-4">
                                    <button type="button" onclick="calculateValuation()" class="btn btn-primary w-100 fs-5 shadow">
                                        ✨ Calculate Property Valuation
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Result Display Card -->
                <div class="col-lg-5">
                    <div class="card p-4 shadow-lg text-center h-100 d-flex flex-column justify-content-center">
                        <span class="label-title mb-2">Estimated Market Valuation</span>
                        <div id="valuation-result" class="val-badge mb-3">$270,314.81</div>
                        
                        <div class="p-3 rounded bg-dark border border-secondary text-start mb-3">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted">Model Engine:</span>
                                <span class="fw-bold text-light">Monotonic XGBoost</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted">Features Ingested:</span>
                                <span class="fw-bold text-light">23 Parameters</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span class="text-muted">Cross-Validation R²:</span>
                                <span class="fw-bold text-success">89.32%</span>
                            </div>
                        </div>
                        
                        <a href="/docs" class="btn btn-outline-light btn-sm mt-2">📖 Open OpenAPI Swagger Docs</a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function calculateValuation() {
                const payload = {
                    "Gr Liv Area": parseFloat(document.getElementById('gr_liv_area').value),
                    "Total_Bsmt_SF": parseFloat(document.getElementById('total_bsmt_sf').value),
                    "Overall Qual": parseInt(document.getElementById('overall_qual').value),
                    "Year Built": parseInt(document.getElementById('year_built').value),
                    "Garage Cars": parseInt(document.getElementById('garage_cars').value),
                    "Full Bath": parseInt(document.getElementById('full_bath').value),
                    "Exter_Qual": document.getElementById('exter_qual').value,
                    "Kitchen_Qual": document.getElementById('kitchen_qual').value
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();
                    if (data.formatted_valuation) {
                        document.getElementById('valuation-result').innerText = data.formatted_valuation;
                    }
                } catch (err) {
                    console.error("Valuation Error:", err);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    model_exists = MODEL_SAVE_PATH.exists()
    return {
        "status": "healthy",
        "model_loaded": model_exists or (_model is not None),
        "model_path": str(MODEL_SAVE_PATH.resolve())
    }

@app.get("/features")
def get_supported_features():
    return {
        "supported_features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS)
    }

@app.post("/predict", response_model=PropertyPredictionResponse)
def predict_valuation(payload: PropertyPredictionRequest):
    try:
        model = get_model()
        input_dict = payload.model_dump(by_alias=True)
        processed_df = preprocess_input_dict(input_dict)
        pred_val = float(model.predict(processed_df)[0])
        
        return PropertyPredictionResponse(
            estimated_valuation=round(pred_val, 2),
            formatted_valuation=f"${pred_val:,.2f}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
