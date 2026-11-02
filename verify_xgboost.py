
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the trained XGBoost model
model = XGBRegressor()
model.load_model("models/xgboost_best.json")

# Load the validation dataset
data = np.load("processed_data/dataset.npz")

X_val = data["X_val"]
y_val = data["y_val"]

# Predict resistance
predictions = model.predict(X_val)

# Calculate validation RMSE
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("Loaded XGBoost model successfully!")
print("Validation RMSE:", f"{rmse:.8f}")
print("First prediction:", f"{predictions[0]:.8f}")

