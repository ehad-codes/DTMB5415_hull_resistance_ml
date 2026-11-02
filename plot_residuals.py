
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Load existing test predictions
with open("results/test_predictions.csv", newline="") as file:
    rows = list(csv.DictReader(file))

actual = np.array([
    float(row["Actual_Cw"]) for row in rows
])

predictions = {
    "Ridge": np.array([float(row["Ridge_Cw"]) for row in rows]),
    "XGBoost": np.array([float(row["XGBoost_Cw"]) for row in rows]),
    "PyTorch MLP": np.array([float(row["MLP_Cw"]) for row in rows])
}

# Plot residuals for all three models
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, predicted) in zip(axes, predictions.items()):

    residuals = predicted - actual

    ax.scatter(actual, residuals, s=8, alpha=0.35)

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        label="Zero error"
    )

    ax.set_title(name)
    ax.set_xlabel("Reference wave-resistance coefficient")
    ax.set_ylabel("Prediction error (Predicted - Actual)")
    ax.set_xlim(0, 0.009)
    ax.set_ylim(-0.006, 0.004)
    ax.grid(alpha=0.2)
    ax.legend()

fig.suptitle(
    "Residual Analysis — Independent Test Set",
    fontsize=14
)

fig.tight_layout()

# Save figure
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "residual_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved results/figures/residual_analysis.png")



