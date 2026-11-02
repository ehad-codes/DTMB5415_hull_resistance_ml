
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)




# Load prepared data
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

# Define the XGBoost regression model
model = XGBRegressor(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    objective="reg:squarederror",
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)

# Train using the training hulls
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)



# Generate validation predictions
predictions = model.predict(X_val)





## PERFORMANCE METRICS


mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\nXGBoost — Validation results")
print("--------------------------------------")
print(f"MAE:  {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²:   {r2:.4f}")

print("\nBest boosting iteration:", model.best_iteration)
print("Best validation RMSE:", model.best_score)

print("\nFirst five predictions:")
for actual, predicted in zip(y_val[:5], predictions[:5]):
    print(
        f"Actual: {actual:.8f} | "
        f"Predicted: {predicted:.8f}"
    )
