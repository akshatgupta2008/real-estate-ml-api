# Ames Real Estate Valuation Pipeline Architecture & Solution Anatomy

## 📌 Business Problem & Domain Context

In real estate valuation, accurate Automated Valuation Models (AVMs) are essential for mortgage underwriting, real estate investment trusts (REITs), and property analytics platforms like Zillow and Redfin. 

The **Ames Housing Dataset** (compiled by Dean De Cock) consists of 2,930 detailed property sales in Ames, Iowa, with 80 raw features spanning structural ratings, living area square footage, neighborhood metrics, and build years.

### The Challenge:
Unconstrained machine learning models often introduce **local prediction inversions** (e.g., predicting a lower dollar valuation for a home with 2,100 sq ft than a home with 2,050 sq ft due to decision tree split noise). 

### Engineering Solution:
We design an end-to-end domain-constrained machine learning pipeline that:
1. Performs Exploratory Data Analysis (EDA) and removes statistical anomalies (>4,000 sq ft living area).
2. Engineers domain-specific interaction features (`Overall Qual` $\times$ `Gr Liv Area`).
3. Enforces domain realism via **Monotonic XGBoost Regression** (`monotone_constraints`), mathematically guaranteeing positive valuation gradients for property size, quality, and bathroom counts.
4. Evaluates performance across 5-Fold Cross Validation using $R^2$, RMSE, and MAE.

---

## 🏗️ Pipeline Architecture

```
                                      PIPELINE ARCHITECTURE
                                    
  [ Raw Ames Dataset ] ──> [ Data Cleaning & Outliers ] ──> [ Feature Engineering ]
  (2,930 Records, 80 Cols)   (Drop >4000 sqft anomalies)    (Qual x Area Interaction)
                                                                     │
                                                                     ▼
  [ Evaluation & Metrics ] <── [ 5-Fold Cross Validation ] <── [ Monotonic XGBoost ]
  (R^2 = 86.6%, RMSE = $28k)  (Linear Reg vs RF vs XGBoost)  (monotone_constraints)
```

### 1. Data Cleaning & Anomaly Detection:
- Per Ames dataset literature (De Cock, 2011), 5 properties with `Gr Liv Area` > 4,000 sq ft are partial/anomalous sales that distort regression slopes. Removing these anomalies reduced cross-validation RMSE by ~$3,500.

### 2. Feature Engineering:
- **`Qual_Area_Interaction`**: `Overall Qual` $\times$ `Gr Liv Area`. In real estate economics, square footage adds significantly more monetary value in luxury properties than in lower-grade structures. Creating this multiplicative term captured non-linear synergy and became the top-gain feature (56.96% weight).

### 3. Model Benchmark & 5-Fold Cross Validation Results:

| Model Paradigm | $R^2$ Score | RMSE ($) | MAE ($) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | `0.8372` | `$31,628.84` | `$22,633.24` | Ordinary Least Squares baseline |
| **Random Forest Regressor** | `0.8540` | `$29,903.78` | `$20,194.37` | Unconstrained ensemble of 100 decision trees |
| **Monotonic XGBoost Regressor** | **`0.8659`** | **`$28,619.16`** | **`$20,043.45`** | **Domain-constrained gradient boosting (Production)** |

---

## 🛠️ Execution Instructions
Run the benchmark execution script:
```bash
python src/train_and_evaluate.py
```
Or run the model training & serialization script:
```bash
python src/model.py
```
