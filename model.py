import pandas as pd
import xgboost as xgb
import joblib

print("Loading AmesHousing dataset...")
# Load the dataset
df = pd.read_csv('AmesHousing.csv')

# 1. Select strong predictive columns
feature_columns = ['Gr Liv Area', 'Bedroom AbvGr', 'Year Built', 'Full Bath']
target_column = 'SalePrice'

# 2. Filter the dataframe and drop any missing values in our chosen columns
data = df[feature_columns + [target_column]].dropna()

X_train = data[feature_columns]
y_train = data[target_column]

print("Training the XGBoost model on real data...")
# 3. Train the model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Save the updated model
joblib.dump(model, 'xgb_model.pkl')
print("Success! New model saved.")