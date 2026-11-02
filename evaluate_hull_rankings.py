
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr




## 1. Load the saved test predictions


with open("results/test_predictions.csv", newline="") as file:
    rows = list(csv.DictReader(file))

froude = np.array([
    float(row["Froude"]) for row in rows
])

actual = np.array([
    float(row["Actual_Cw"]) for row in rows
])

predictions = {
    "Ridge": np.array([
        float(row["Ridge_Cw"]) for row in rows
    ]),
    "XGBoost": np.array([
        float(row["XGBoost_Cw"]) for row in rows
    ]),
    "PyTorch MLP": np.array([
        float(row["MLP_Cw"]) for row in rows
    ])
}




## 2. Calculate ranking correlations


results = []

print("\nHull-design ranking comparison")
print("------------------------------------------")

for fr in [0.25, 0.33, 0.41]:

    mask = np.isclose(froude, fr)

    reference_cw = actual[mask]

    print(f"\nFroude number: {fr:.2f}")
    print(f"Hulls evaluated: {mask.sum()}")

    for name, predicted in predictions.items():

        predicted_cw = predicted[mask]

        rho, p_value = spearmanr(
            reference_cw,
            predicted_cw
        )

        results.append({
            "Froude": fr,
            "Model": name,
            "Spearman_rho": rho
        })

        print(f"{name:<12} | Spearman rho: {rho:.4f}")






## 3. Save the results


output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "ranking_metrics.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["Froude", "Model", "Spearman_rho"]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nSaved results/ranking_metrics.csv")
