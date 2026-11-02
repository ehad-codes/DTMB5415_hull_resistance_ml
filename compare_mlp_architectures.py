
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Load prepared data
data = np.load("processed_data/dataset.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
X_val = torch.tensor(data["X_val"], dtype=torch.float32)

y_train = data["y_train"]
y_val = data["y_val"]



# Scale targets using training data only
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(
    y_train.reshape(-1, 1)
)
y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
)



y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train_tensor)

class ResistanceMLP(nn.Module):
    def __init__(self, hidden1, hidden2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.network(x)

architectures = [(32, 16), (64, 32), (128, 64)]
results = []

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

for hidden1, hidden2 in architectures:

    # Reinitialise random state for each experiment
    torch.manual_seed(42)
    np.random.seed(42)

    model = ResistanceMLP(hidden1, hidden2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_epoch = None
    best_weights = None
    patience_counter = 0

    for epoch in range(100):

        model.train()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_loss = criterion(
                model(X_val), y_val_tensor
            ).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break



    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Predict and convert back to original Cw scale
    with torch.no_grad():
        pred_scaled = model(X_val).numpy()

    pred = target_scaler.inverse_transform(
        pred_scaled
    ).flatten()

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    # Save model with architecture-specific filename
    filename = f"mlp_{hidden1}_{hidden2}.pth"
    torch.save(model.state_dict(), model_dir / filename)

    results.append((hidden1, hidden2, rmse, r2))

    print(
        f"{hidden1} → {hidden2} | "
        f"RMSE: {rmse:.8f} | R²: {r2:.4f} | "
        f"Best epoch: {best_epoch}"
    )

print("\nArchitecture comparison")
print("--------------------------------------")

for h1, h2, rmse, r2 in results:
    print(
        f"{h1:3d} → {h2:3d} | "
        f"RMSE: {rmse:.8f} | R²: {r2:.4f}"
    )
