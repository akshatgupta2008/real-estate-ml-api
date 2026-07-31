import pandas as pd
import xgboost as xgb
import joblib
from sklearn.datasets import make_regression

print("Training the AI model...")

# Creating fake data just for testing today
X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
X_train = pd.DataFrame(X, columns=['sqft', 'bedrooms', 'age'])
y_train = pd.Series(y)

# Train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'xgb_model.pkl')
print("Success! Model saved.")