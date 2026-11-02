
from pathlib import Path

import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load the existing processed dataset
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

# Train the selected Ridge configuration
model = Ridge(alpha=100)

model.fit(X_train, y_train)

# Evaluate using validation data only
predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("Tuned Ridge — Validation results")
print("--------------------------------------")
print(f"MAE:  {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²:   {r2:.4f}")

# Save the trained model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "ridge_best.joblib")

print("\nRidge model saved successfully!")
