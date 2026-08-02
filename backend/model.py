import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

DATASET_PATH = Path(__file__).parent / 'AmesHousing.csv'
MODEL_PATH = Path(__file__).parent / 'xgb_model.pkl'

print("Loading AmesHousing dataset...")
df = pd.read_csv(DATASET_PATH)

feature_columns = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Full Bath', 'Bedroom AbvGr', 'Year Built']
target_column = 'SalePrice'

data = df[feature_columns + [target_column]].dropna()

X_train = data[feature_columns]
y_train = data[target_column]

print("Training high-accuracy XGBoost model with monotonicity constraints...")
# monotone_constraints enforce that higher quality, larger area, more garage cars,
# more bathrooms, and newer construction year monotonically increase property value.
model = xgb.XGBRegressor(
    n_estimators=180,
    learning_rate=0.07,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    monotone_constraints='(1, 1, 1, 1, 0, 1)',
    random_state=42
)
model.fit(X_train, y_train)

r2_score = model.score(X_train, y_train)
print(f"Model Training Complete! R^2 Score: {r2_score:.4f}")

joblib.dump(model, MODEL_PATH)
print("Saved retrained xgb_model.pkl successfully!")