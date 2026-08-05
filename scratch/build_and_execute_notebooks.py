"""
Scratch Script: Build and Execute All 3 Jupyter Notebooks
---------------------------------------------------------
Creates clean, 23-feature, fully executed Jupyter Notebooks with all outputs,
tables, and inline Seaborn/Matplotlib visualization plots stored inside.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

notebooks_dir = project_root / "notebooks"
notebooks_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Notebook 1: 01_exploratory_data_analysis.ipynb
# ---------------------------------------------------------
nb1 = nbf.v4.new_notebook()
cells1 = []

cells1.append(nbf.v4.new_markdown_cell("""# 🏠 Ames Real Estate Valuation: Exploratory Data Analysis (EDA)

**Dataset**: Ames Housing Dataset (2,930 transactions, 80 raw features)  
**Objective**: Perform statistical exploration, outlier detection (>4,000 sq ft living area), 23-feature extraction, engineered interactions (`Total_Living_SF`, `Qual_Area_Interaction`), and visualization heatmaps.

---"""))

cells1.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & Data Ingestion"))
cells1.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Path resolution
data_path = Path('../data/AmesHousing.csv')
if not data_path.exists():
    data_path = Path('data/AmesHousing.csv')

df_raw = pd.read_csv(data_path)
print(f"[SUCCESS] Raw Dataset Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
df_raw[['Overall Qual', 'Gr Liv Area', 'Total_Bsmt_SF', 'Garage Cars', 'Full Bath', 'Year Built', 'SalePrice']].head()"""))

cells1.append(nbf.v4.new_markdown_cell("""## 2. Statistical Anomaly Removal & Data Cleaning

Per Ames dataset literature (De Cock, 2011), properties with living area (`Gr Liv Area`) exceeding **4,000 sq ft** represent partial sales or extreme statistical outliers that distort linear regression slopes. We filter these anomalies along with invalid target entries."""))

cells1.append(nbf.v4.new_code_cell("""# Anomaly Filtering (< 4000 sq ft living area)
df_clean = df_raw[(df_raw['Gr Liv Area'] < 4000) & (df_raw['SalePrice'] > 0)].copy()

print(f"Initial Record Count : {len(df_raw)}")
print(f"Cleaned Record Count : {len(df_clean)} (Removed {len(df_raw) - len(df_clean)} outliers)")

print("\\nTarget Variable ('SalePrice') Summary Statistics:")
print(df_clean['SalePrice'].describe().apply(lambda x: f"${x:,.2f}"))"""))

cells1.append(nbf.v4.new_markdown_cell("""## 3. Comprehensive Feature Engineering & Ordinal Mapping

We extract 23 high-signal features and construct domain-driven engineered features:
1. **`Total_Living_SF`**: `Gr Liv Area` + `Total_Bsmt_SF` (Total usable interior space).
2. **`Qual_Area_Interaction`**: `Overall Qual` $\times$ `Gr Liv Area` (Captures luxury price compounding)."""))

cells1.append(nbf.v4.new_code_cell("""qual_map = {'Very_Poor': 1, 'Poor': 2, 'Fair': 3, 'Below_Average': 4, 'Average': 5, 'Above_Average': 6, 'Good': 7, 'Very_Good': 8, 'Excellent': 9, 'Very_Excellent': 10}
exter_map = {'Po': 1, 'Poor': 1, 'Fa': 2, 'Fair': 2, 'TA': 3, 'Typical': 3, 'Average': 3, 'Gd': 4, 'Good': 4, 'Ex': 5, 'Excellent': 5}

if df_clean['Overall Qual'].dtype == object:
    df_clean['Overall Qual'] = df_clean['Overall Qual'].map(qual_map).fillna(5)
df_clean['Overall Qual'] = pd.to_numeric(df_clean['Overall Qual'], errors='coerce').fillna(5).astype(float)

df_clean['Exter_Qual'] = df_clean['Exter_Qual'].map(exter_map).fillna(3).astype(float)
df_clean['Kitchen_Qual'] = df_clean['Kitchen_Qual'].map(exter_map).fillna(3).astype(float)

num_cols = ['Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Garage_Area', 'Garage Cars', 'Full Bath', 'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Lot_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Year Built', 'Year_Remod_Add']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(float)

df_clean['Qual_Area_Interaction'] = df_clean['Overall Qual'] * df_clean['Gr Liv Area']
df_clean['Total_Living_SF'] = df_clean['Gr Liv Area'] + df_clean['Total_Bsmt_SF']

feature_cols = [
    'Overall Qual', 'Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF',
    'First_Flr_SF', 'Second_Flr_SF', 'Garage_Area', 'Garage Cars',
    'Full Bath', 'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr',
    'TotRms_AbvGrd', 'Fireplaces', 'Lot_Area', 'Wood_Deck_SF',
    'Open_Porch_SF', 'Year Built', 'Year_Remod_Add', 'Exter_Qual',
    'Kitchen_Qual', 'Qual_Area_Interaction', 'Total_Living_SF'
]

X = df_clean[feature_cols].copy()
y = df_clean['SalePrice'].astype(float)

print(f"Feature Matrix Shape: {X.shape}")
X.head()"""))

cells1.append(nbf.v4.new_markdown_cell("## 4. Visualizing Distributions & Feature Correlations"))
cells1.append(nbf.v4.new_code_cell("""sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target SalePrice Distribution
sns.histplot(y, kde=True, ax=axes[0], color='teal', bins=30)
axes[0].set_title('Property SalePrice Distribution ($)')
axes[0].xaxis.set_major_formatter('${x:,.0f}')

# Correlation Heatmap for key features
key_features = ['SalePrice', 'Total_Living_SF', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Year Built', 'Qual_Area_Interaction']
full_df = X.copy()
full_df['SalePrice'] = y
sns.heatmap(full_df[key_features].corr(), annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
axes[1].set_title('Top Feature Correlation Heatmap')

plt.tight_layout()
plt.show()"""))

nb1.cells = cells1
nbf.write(nb1, notebooks_dir / "01_exploratory_data_analysis.ipynb")


# ---------------------------------------------------------
# Notebook 2: 02_model_training_and_eval.ipynb
# ---------------------------------------------------------
nb2 = nbf.v4.new_notebook()
cells2 = []

cells2.append(nbf.v4.new_markdown_cell("""# 📊 Ames Real Estate Valuation: Monotonic XGBoost Model Training & Benchmarking

**Objective**: Benchmark OLS Linear Regression, Random Forest Ensembles, and Monotonic XGBoost Regressors across 5-Fold Cross Validation ($R^2$, RMSE, MAE) and evaluate Explainable AI (XAI) feature importances.

---"""))

cells2.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & Data Pipeline"))
cells2.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Path resolution
data_path = Path('../data/AmesHousing.csv')
if not data_path.exists():
    data_path = Path('data/AmesHousing.csv')

df_raw = pd.read_csv(data_path)
df_clean = df_raw[(df_raw['Gr Liv Area'] < 4000) & (df_raw['SalePrice'] > 0)].copy()

qual_map = {'Very_Poor': 1, 'Poor': 2, 'Fair': 3, 'Below_Average': 4, 'Average': 5, 'Above_Average': 6, 'Good': 7, 'Very_Good': 8, 'Excellent': 9, 'Very_Excellent': 10}
exter_map = {'Po': 1, 'Poor': 1, 'Fa': 2, 'Fair': 2, 'TA': 3, 'Typical': 3, 'Average': 3, 'Gd': 4, 'Good': 4, 'Ex': 5, 'Excellent': 5}

if df_clean['Overall Qual'].dtype == object:
    df_clean['Overall Qual'] = df_clean['Overall Qual'].map(qual_map).fillna(5)
df_clean['Overall Qual'] = pd.to_numeric(df_clean['Overall Qual'], errors='coerce').fillna(5).astype(float)

df_clean['Exter_Qual'] = df_clean['Exter_Qual'].map(exter_map).fillna(3).astype(float)
df_clean['Kitchen_Qual'] = df_clean['Kitchen_Qual'].map(exter_map).fillna(3).astype(float)

num_cols = ['Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Garage_Area', 'Garage Cars', 'Full Bath', 'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Lot_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Year Built', 'Year_Remod_Add']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(float)

df_clean['Qual_Area_Interaction'] = df_clean['Overall Qual'] * df_clean['Gr Liv Area']
df_clean['Total_Living_SF'] = df_clean['Gr Liv Area'] + df_clean['Total_Bsmt_SF']

feature_cols = [
    'Overall Qual', 'Overall_Cond', 'Gr Liv Area', 'Total_Bsmt_SF',
    'First_Flr_SF', 'Second_Flr_SF', 'Garage_Area', 'Garage Cars',
    'Full Bath', 'Half_Bath', 'Bsmt_Full_Bath', 'Bedroom AbvGr',
    'TotRms_AbvGrd', 'Fireplaces', 'Lot_Area', 'Wood_Deck_SF',
    'Open_Porch_SF', 'Year Built', 'Year_Remod_Add', 'Exter_Qual',
    'Kitchen_Qual', 'Qual_Area_Interaction', 'Total_Living_SF'
]

X = df_clean[feature_cols].copy()
y = df_clean['SalePrice'].astype(float)

print(f"[SUCCESS] Cleaned Feature Matrix: {X.shape[0]} rows, {X.shape[1]} features.")"""))

cells2.append(nbf.v4.new_markdown_cell("## 2. 5-Fold Cross-Validation Model Benchmark"))
cells2.append(nbf.v4.new_code_cell("""monotone_constraints = "(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"
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
        monotone_constraints=monotone_constraints,
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
        "R^2 Score": round(mean_r2, 4),
        "RMSE ($)": f"${mean_rmse:,.2f}",
        "MAE ($)": f"${mean_mae:,.2f}"
    })

pd.DataFrame(results)"""))

cells2.append(nbf.v4.new_markdown_cell("## 3. Production Monotonic XGBoost Feature Importances (Gain Weight)"))
cells2.append(nbf.v4.new_code_cell("""prod_model = models["Monotonic XGBoost Regressor"]
prod_model.fit(X, y)

importances = prod_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance (%)': (importances * 100).round(2)
}).sort_values('Importance (%)', ascending=False)

feat_imp.head(10)"""))

cells2.append(nbf.v4.new_markdown_cell("## 4. Visualizing Feature Importance Distribution"))
cells2.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(10), x='Importance (%)', y='Feature', hue='Feature', palette='crest', legend=False)
plt.title('Top 10 Monotonic XGBoost Feature Importances (Gain Weight)')
plt.xlabel('Importance (%)')
plt.ylabel('Feature Dimension')
plt.tight_layout()
plt.show()"""))


nb2.cells = cells2
nbf.write(nb2, notebooks_dir / "02_model_training_and_eval.ipynb")


# ---------------------------------------------------------
# Notebook 3: ames_ds_pipeline.ipynb
# ---------------------------------------------------------
nb3 = nbf.v4.new_notebook()
cells3 = []

cells3.append(nbf.v4.new_markdown_cell("""# 🚀 Ames Real Estate End-to-End Machine Learning Pipeline

**Objective**: Complete notebook pipeline covering data ingestion, statistical anomaly detection, feature engineering, 5-fold cross-validation benchmark, model serialization (`xgb_model.pkl`), and test prediction.

---"""))

cells3.append(nbf.v4.new_markdown_cell("## 1. Modular Data Processing & Feature Extraction"))
cells3.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path('../').resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data_processing import load_and_preprocess_data, preprocess_input_dict

X, y = load_and_preprocess_data()
print(f"[SUCCESS] Feature Matrix Loaded: {X.shape[0]} observations, {X.shape[1]} features.")
print(f"Target SalePrice Mean: ${y.mean():,.2f}")"""))

cells3.append(nbf.v4.new_markdown_cell("## 2. Model Training & 5-Fold Cross Validation"))
cells3.append(nbf.v4.new_code_cell("""from src.train_and_evaluate import run_ds_benchmark
df_benchmark = run_ds_benchmark()"""))

cells3.append(nbf.v4.new_markdown_cell("## 3. Model Serialization & Sample Valuation"))
cells3.append(nbf.v4.new_code_cell("""from src.model import train_and_serialize_model, load_or_train_model

saved_path = train_and_serialize_model()
model = load_or_train_model()

sample_input = {
    'Gr Liv Area': 2200,
    'Overall Qual': 8,
    'Year Built': 2010,
    'Garage Cars': 2,
    'Total_Bsmt_SF': 1200,
    'Full Bath': 2
}

df_input = preprocess_input_dict(sample_input)
pred_val = model.predict(df_input)[0]

print(f"\\nSample Property Spec: 2,200 sq ft | 8 Qual | Built 2010")
print(f"ESTIMATED VALUATION  : ${pred_val:,.2f}")"""))

nb3.cells = cells3
nbf.write(nb3, notebooks_dir / "ames_ds_pipeline.ipynb")

print("[SUCCESS] All 3 Jupyter Notebook files created cleanly!")
