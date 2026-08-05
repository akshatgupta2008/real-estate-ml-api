# Ames Real Estate Valuation & Monotonic XGBoost ML Engine

An end-to-end Data Science and Machine Learning project that predicts single-family real estate values with domain-constrained **Monotonic XGBoost Regression** ($R^2 = 86.6\%$) and provides Explainable AI (XAI) feature interpretability.

---

## 📌 Project Overview & Data Science Objectives

Predicting property values with unconstrained machine learning models often introduces **local prediction inversions** (e.g., predicting a lower valuation for a 2,100 sq ft home than a 2,050 sq ft home due to tree split noise). 

This project addresses this by engineering a **domain-constrained Machine Learning pipeline**:
* **Outlier Removal & Data Cleaning**: Removes statistical anomalies (>4,000 sq ft living area as recommended in Ames dataset literature).
* **Feature Engineering**: Constructs interaction features like `Quality x Area Interaction` (`Overall Qual` $\times$ `Gr Liv Area`) to model non-linear price compounding in high-grade properties.
* **Monotonic XGBoost Regression**: Enforces monotonic constraint vectors (`monotone_constraints`) so valuation scales logically with property size, build quality, and bathroom count.
* **Rigorous Validation**: Evaluates baseline Linear Regression, Random Forest, and Monotonic XGBoost across 5-fold cross-validation.
* **Model Serialization**: Fits and serializes the production XGBoost model (`xgb_model.pkl`) with `joblib`.

---

## 📊 Model Benchmark & Performance Metrics

Evaluated across **5-Fold Cross Validation** on 2,924 cleaned Ames housing records:

| Model Paradigm | $R^2$ Score | RMSE ($) | MAE ($) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | `0.8372` | `$31,628.84` | `$22,633.24` | Ordinary Least Squares baseline |
| **Random Forest Regressor** | `0.8540` | `$29,903.78` | `$20,194.37` | Unconstrained ensemble of 100 decision trees |
| **Monotonic XGBoost Regressor** | **`0.8659`** | **`$28,619.16`** | **`$20,043.45`** | **Domain-constrained gradient boosting (Production)** |

---

## 📈 Feature Importance & Explainable AI (XAI)

XGBoost Gain Feature Importance Breakdown:
* **Quality x Area Interaction** (`56.96%`): Single largest predictor capturing non-linear quality-square footage synergy.
* **Overall Quality Rating** (`26.22%`): Primary structural rating dimension (1–10).
* **Garage Capacity (Cars)** (`8.57%`): Garage vehicle capacity.
* **Year Built** (`4.41%`): Construction year.
* **Living Area (sq ft)** (`2.11%`): Ground living square footage.
* **Bedrooms & Bathrooms** (`1.74%`): Room metrics.

---

## 📁 Repository Layout

```
real-estate-ml-api/
├── data/                         # Ames Housing Dataset
│   └── AmesHousing.csv           # Ames Housing dataset (2,930 records)
│
├── notebooks/                    # Data Science EDA & Model Benchmark Notebooks
│   ├── 01_exploratory_data_analysis.ipynb # EDA, Pandas summary stats & Seaborn visualizations
│   └── 02_model_training_and_eval.ipynb  # Monotonic XGBoost training & 5-fold CV evaluation
│
├── src/                          # Modular Python Machine Learning Engine
│   ├── __init__.py
│   ├── data_processing.py        # Data cleaning, outlier removal (>4000 sq ft) & feature engineering
│   ├── train_and_evaluate.py     # 5-Fold cross-validation benchmark script
│   └── model.py                  # XGBoost training & joblib serialization pipeline
│
├── docs/                         # Technical Documentation & Interview Preparation
│   ├── real_estate_valuation_anatomy.md # Problem statement, business context & architecture
│   └── ds_interview_cheat_sheet.md      # Resume bullet points, pitches & technical Q&A
│
├── requirements.txt              # Python ML dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Science Benchmark Script:
```bash
python src/train_and_evaluate.py
```

### 3. Train & Serialize Monotonic XGBoost Model:
```bash
python src/model.py
```
