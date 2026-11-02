
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the existing prepared datasets
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

learning_rates = [0.01, 0.03, 0.05, 0.1]

print("XGBoost learning-rate comparison")
print("--------------------------------------")

results = []

for lr in learning_rates:

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=lr,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

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

    results.append((lr, mae, rmse, r2, model.best_iteration))

    print(
        f"LR: {lr:<5} | "
        f"MAE: {mae:.8f} | "
        f"RMSE: {rmse:.8f} | "
        f"R²: {r2:.4f} | "
        f"Best iteration: {model.best_iteration}"
    )

best = min(results, key=lambda row: row[2])

print("\nBest learning rate:", best[0])
print(f"Best validation RMSE: {best[2]:.8f}")
