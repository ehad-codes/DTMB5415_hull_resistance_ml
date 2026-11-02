
from pathlib import Path

import joblib
import numpy as np

import torch

from torch import nn
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


## 1. Load the untouched test dataset


data = np.load("processed_data/dataset.npz")

X_test = data["X_test"]
y_test = data["y_test"]




## 2. Define the selected PyTorch architecture


class ResistanceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)



## 3. Load the saved models


ridge = joblib.load("models/ridge_best.joblib")

xgboost = XGBRegressor()
xgboost.load_model("models/xgboost_best.json")

mlp = ResistanceMLP()
mlp.load_state_dict(
    torch.load(
        "models/mlp_128_64.pth",
        map_location="cpu",
        weights_only=True
    )
)
mlp.eval()

target_scaler = joblib.load(
    "models/target_scaler.joblib"
)


## 4. Generate predictions


ridge_pred = ridge.predict(X_test)

xgboost_pred = xgboost.predict(X_test)

with torch.no_grad():
    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    mlp_pred_scaled = mlp(
        X_test_tensor
    ).numpy()

mlp_pred = target_scaler.inverse_transform(
    mlp_pred_scaled
).flatten()




## 5. Calculate test metrics


def evaluate_model(name, actual, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

results = [
    evaluate_model("Ridge", y_test, ridge_pred),
    evaluate_model("XGBoost", y_test, xgboost_pred),
    evaluate_model("PyTorch MLP", y_test, mlp_pred)
]




## 6. Display and save results


import csv

print("\nFinal test-set comparison")
print("--------------------------------------")
print(f"{'Model':<15} {'MAE':>12} {'RMSE':>12} {'R²':>10}")

for result in results:
    print(
        f"{result['Model']:<15} "
        f"{result['MAE']:>12.8f} "
        f"{result['RMSE']:>12.8f} "
        f"{result['R2']:>10.4f}"
    )

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "test_metrics.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Model", "MAE", "RMSE", "R2"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to results/test_metrics.csv")






## 7. Evaluate performance at each Froude number


froude_numbers = [0.25, 0.33, 0.41]

# The 28th input feature is the scaled Froude number.
# Recover the original values using the fitted input scaler.
feature_scaler = joblib.load(
    "processed_data/feature_scaler.joblib"
)

froude_mean = feature_scaler.mean_[-1]
froude_std = feature_scaler.scale_[-1]

froude_test = X_test[:, -1] * froude_std + froude_mean

models_predictions = {
    "Ridge": ridge_pred,
    "XGBoost": xgboost_pred,
    "PyTorch MLP": mlp_pred
}

froude_results = []

print("\nPerformance by Froude number")
print("--------------------------------------")

for fr in froude_numbers:

    mask = np.isclose(froude_test, fr)

    print(f"\nFr = {fr:.2f} | Observations: {mask.sum()}")

    for name, predictions in models_predictions.items():

        mae = mean_absolute_error(
            y_test[mask], predictions[mask]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test[mask], predictions[mask]
            )
        )

        r2 = r2_score(
            y_test[mask], predictions[mask]
        )

        froude_results.append({
            "Froude": fr,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"{name:<12} | "
            f"MAE: {mae:.8f} | "
            f"RMSE: {rmse:.8f} | "
            f"R²: {r2:.4f}"
        )

# Save Froude-specific results
with open(
    output_dir / "froude_metrics.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["Froude", "Model", "MAE", "RMSE", "R2"]
    )

    writer.writeheader()
    writer.writerows(froude_results)

print("\nSaved results/froude_metrics.csv")








## 8. Save individual predictions and errors


prediction_file = output_dir / "test_predictions.csv"

with open(prediction_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Froude",
        "Actual_Cw",
        "Ridge_Cw",
        "XGBoost_Cw",
        "MLP_Cw",
        "Ridge_Error",
        "XGBoost_Error",
        "MLP_Error"
    ])

    for i in range(len(y_test)):
        writer.writerow([
            froude_test[i],
            y_test[i],
            ridge_pred[i],
            xgboost_pred[i],
            mlp_pred[i],
            ridge_pred[i] - y_test[i],
            xgboost_pred[i] - y_test[i],
            mlp_pred[i] - y_test[i]
        ])

print(f"\nSaved {prediction_file}")










## 9. Identify largest absolute errors for each model


for name, predictions in models_predictions.items():

    absolute_errors = np.abs(predictions - y_test)

    # Indices of the five largest errors
    worst_indices = np.argsort(absolute_errors)[-5:][::-1]

    print(f"\nFive largest errors — {name}")
    print("--------------------------------------")

    for i in worst_indices:
        print(
            f"Test index: {i:4d} | "
            f"Fr: {froude_test[i]:.2f} | "
            f"Actual: {y_test[i]:.8f} | "
            f"Predicted: {predictions[i]:.8f} | "
            f"Abs error: {absolute_errors[i]:.8f}"
        )
