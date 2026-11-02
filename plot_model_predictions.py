
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Load the saved test predictions
with open("results/test_predictions.csv", newline="") as file:
    rows = list(csv.DictReader(file))

actual = np.array([float(row["Actual_Cw"]) for row in rows])

predictions = {
    "Ridge": np.array([float(row["Ridge_Cw"]) for row in rows]),
    "XGBoost": np.array([float(row["XGBoost_Cw"]) for row in rows]),
    "PyTorch MLP": np.array([float(row["MLP_Cw"]) for row in rows])
}

# Create a figure with one panel per model
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, predicted) in zip(axes, predictions.items()):

    ax.scatter(
        actual,
        predicted,
        s=8,
        alpha=0.35
    )

    # Ideal prediction line
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        label="Perfect prediction"
    )

    ax.set_title(name)
    ax.set_xlim(0, 0.009)
    ax.set_ylim(0, 0.009)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference wave-resistance coefficient")
    ax.set_ylabel("Predicted wave-resistance coefficient")
    ax.legend()
    ax.grid(alpha=0.2)

fig.suptitle(
    "Hull Resistance Prediction — Independent Test Set",
    fontsize=14
)

fig.tight_layout()

# Save the plot
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "predicted_vs_reference.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved results/figures/predicted_vs_reference.png")
