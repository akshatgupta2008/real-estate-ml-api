# Ames Real Estate Valuation Pipeline Architecture & Solution Anatomy

## 📌 Business Problem & Domain Context

In real estate valuation, accurate Automated Valuation Models (AVMs) are essential for mortgage underwriting, real estate investment trusts (REITs), and property analytics platforms like Zillow and Redfin. 

The **Ames Housing Dataset** (compiled by Dean De Cock) consists of 2,930 detailed property sales in Ames, Iowa, with 80 raw features spanning structural ratings, living area square footage, neighborhood metrics, and build years.

### The Challenge:
Unconstrained machine learning models often introduce **local prediction inversions** (e.g., predicting a lower dollar valuation for a home with 2,100 sq ft than a home with 2,050 sq ft due to decision tree split noise). 

### Engineering Solution:
We design an end-to-end domain-constrained machine learning pipeline and API that:
1. Performs Exploratory Data Analysis (EDA) and removes statistical anomalies (>4,000 sq ft living area).
2. Extracts 23 comprehensive structural, quality, area, basement, garage, and room features.
3. Engineers domain-specific interaction features (`Overall Qual` $\times$ `Gr Liv Area` and `Gr Liv Area` + `Total_Bsmt_SF`).
4. Enforces domain realism via **Monotonic XGBoost Regression** (`monotone_constraints`), mathematically guaranteeing positive valuation gradients for property size, quality, and bathroom counts.
5. Evaluates performance across 5-Fold Cross Validation using $R^2$, RMSE, and MAE.
6. Exposes real-time predictions via an interactive CLI (`src/predict.py`) and FastAPI web server (`src/api.py`).

---

## 🏗️ Pipeline Architecture

```
                                      PIPELINE ARCHITECTURE
                                    
  [ Raw Ames Dataset ] ──> [ Data Cleaning & Imputation ] ──> [ Feature Engineering ]
  (2,930 Records, 80 Cols)   (Drop >4000 sqft anomalies)    (23 Features, Total SF, Qual Interaction)
                                                                      │
                                                                      ▼
  [ REST API / CLI ] <── [ Model Serialization ] <── [ Monotonic XGBoost ] <── [ 5-Fold Cross Validation ]
  (FastAPI / predict.py)   (xgb_model.pkl via joblib) (monotone_constraints)    (R^2 = 89.3%, RMSE = $25k)
```

### 1. Data Cleaning & Anomaly Detection:
- Per Ames dataset literature (De Cock, 2011), properties with `Gr Liv Area` > 4,000 sq ft represent partial/anomalous sales that distort regression slopes.

### 2. Feature Engineering:
- **`Total_Living_SF`**: `Gr Liv Area` + `Total_Bsmt_SF`. Combines above-ground living area with finished basement space, becoming the single top predictor (40.90% gain weight).
- **`Qual_Area_Interaction`**: `Overall Qual` $\times$ `Gr Liv Area`. Captures non-linear luxury price amplification.

### 3. Model Benchmark & 5-Fold Cross Validation Results:

| Model Paradigm | $R^2$ Score | RMSE ($) | MAE ($) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | `0.8165` | `$33,535.06` | `$23,485.40` | Ordinary Least Squares baseline |
| **Random Forest Regressor** | `0.8858` | `$26,397.04` | `$17,504.55` | Ensemble of 150 decision trees |
| **Monotonic XGBoost Regressor** | **`0.8932`** | **`$25,403.24`** | **`$17,433.08`** | **Domain-constrained gradient boosting (Production)** |

---

## 🛠️ Execution Instructions
Run the benchmark execution script:
```bash
python src/train_and_evaluate.py
```
Train & serialize the production model:
```bash
python src/model.py
```
Test property valuation interactively:
```bash
python src/predict.py --sample
```
Launch FastAPI REST server:
```bash
uvicorn src.api:app --reload
```
