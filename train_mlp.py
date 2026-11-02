## DEFINING HYPERPARAMTERS AND NEURAL NETWORK ARCHITECTURE 


import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load our prepared dataset
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
X_val = data["X_val"]

y_train = data["y_train"]
y_val = data["y_val"]

# Standardise the resistance target using training data only
target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.reshape(-1, 1)
)

y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
)

# Convert NumPy arrays into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)





# Define our neural network

class ResistanceMLP(nn.Module):
    def __init__(self, hidden1=64, hidden2=32):
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




model = ResistanceMLP(hidden1=32, hidden2=16)

print("Neural network architecture:")
print(model)

print("\nTraining input shape:", X_train_tensor.shape)
print("Training target shape:", y_train_tensor.shape)

print("\nTarget scaling:")
print("Training mean Cw:", target_scaler.mean_[0])
print("Training standard deviation Cw:", target_scaler.scale_[0])

print("\nTotal trainable parameters:",
      sum(p.numel() for p in model.parameters()))











## TRAINING MLP



from torch.utils.data import TensorDataset, DataLoader
import copy

# Combine inputs and targets into a training dataset
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

# Divide training observations into mini-batches
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# Define the loss function and optimiser
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training settings
epochs = 100
patience = 15

best_val_loss = float("inf")
best_weights = None
epochs_without_improvement = 0

train_losses = []
val_losses = []

for epoch in range(epochs):

    # TRAINING PHASE
    model.train()

    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Reset gradients from the previous batch
        optimizer.zero_grad()

        # Predict resistance
        predictions = model(X_batch)

        # Calculate prediction error
        loss = criterion(predictions, y_batch)

        # Calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate squared-error sum
        total_train_loss += loss.item() * len(X_batch)

    avg_train_loss = total_train_loss / len(train_dataset)




    # VALIDATION PHASE
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(
            val_predictions,
            y_val_tensor
        ).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)



    # Save the best-performing network weights
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train loss: {avg_train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )

    # Stop if validation performance stops improving
    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break


# Restore the weights with the lowest validation loss
model.load_state_dict(best_weights)

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.6f}")






## PERFORMANCE METRICS



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Switch to evaluation mode
model.eval()

# Generate predictions for validation hulls
with torch.no_grad():
    predictions_scaled = model(X_val_tensor).numpy()

# Convert predictions back to original Cw values
predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

# Calculate evaluation metrics using original Cw values
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\nPyTorch MLP — Validation results")
print("--------------------------------------")
print(f"MAE:  {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²:   {r2:.4f}")

print("\nFirst five predictions:")
for actual, predicted in zip(y_val[:5], predictions[:5]):
    print(
        f"Actual: {actual:.8f} | "
        f"Predicted: {predicted:.8f}"
    )





## SAVING BEST NEURAL NETWORK WEIGHTS



from pathlib import Path
import joblib

# Create a directory for trained models
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Save the best neural-network weights
torch.save(
    model.state_dict(),
    model_dir / "mlp_baseline.pth"
)

# Save the target scaler used during training
joblib.dump(
    target_scaler,
    model_dir / "target_scaler.joblib"
)

print("\nModel and target scaler saved successfully!")
