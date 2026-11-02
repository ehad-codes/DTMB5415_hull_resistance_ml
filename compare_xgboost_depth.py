
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared data
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

# Compare different tree depths
depths = [3, 4, 6, 8]

results = []

print("XGBoost hyperparameter comparison")
print("--------------------------------------")

for depth in depths:

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=depth,
        learning_rate=0.05,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    )

    # Train on the same hull configurations
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append((depth, mae, rmse, r2, model.best_iteration))

    print(
        f"Depth {depth} | "
        f"MAE: {mae:.8f} | "
        f"RMSE: {rmse:.8f} | "
        f"R²: {r2:.4f} | "
        f"Best iteration: {model.best_iteration}"
    )

print("\nComparison summary")
print("--------------------------------------")

for depth, mae, rmse, r2, iteration in results:
    print(
        f"Depth {depth}: RMSE = {rmse:.8f}, "
        f"R² = {r2:.4f}"
    )
