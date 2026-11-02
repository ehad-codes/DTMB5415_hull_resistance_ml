
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

# Load prepared training and validation data
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

# Selected hyperparameters
model = XGBRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    objective="reg:squarederror",
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

# Train and select best boosting iteration
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Save model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model.save_model(
    model_dir / "xgboost_best.json"
)

print("XGBoost model saved successfully!")
print("Best iteration:", model.best_iteration)
print("Best validation RMSE:", model.best_score)
