##RIDGE REGRESSION

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load our previously prepared dataset
data = np.load("processed_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]


# Create the Ridge regression model
model = Ridge(alpha=1.0)


# Train the model using the training dataset
model.fit(X_train, y_train)


# Predict resistance for validation hulls
y_pred = model.predict(X_val)


# Calculate validation performance
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)



print("Ridge regression — Validation results")
print("--------------------------------------")
print(f"MAE:  {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²:   {r2:.4f}")

print("\nFirst five predictions:")
for actual, predicted in zip(y_val[:5], y_pred[:5]):
    print(f"Actual: {actual:.8f} | Predicted: {predicted:.8f}")




## OPTIMISING ALPHA HYPERPARAMETER




# Test different regularisation strengths
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

best_alpha = None
best_rmse = float("inf")


print("\nRidge hyperparameter tuning")
print("--------------------------------------")

for alpha in alphas:

    # Create and train a new model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    # Predict on validation data
    predictions = ridge.predict(X_val)

    # Calculate validation error
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    print(f"Alpha: {alpha:<8} RMSE: {rmse:.8f}")

    # Keep track of the best-performing alpha
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print("\nBest alpha:", best_alpha)
print(f"Best validation RMSE: {best_rmse:.8f}")















## SIMPLE MEAN MODEL




from sklearn.dummy import DummyRegressor

# Baseline that always predicts the training mean
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)

dummy_pred = dummy.predict(X_val)

dummy_mae = mean_absolute_error(y_val, dummy_pred)
dummy_rmse = np.sqrt(mean_squared_error(y_val, dummy_pred))
dummy_r2 = r2_score(y_val, dummy_pred)

print("\nMean predictor — Validation results")
print("--------------------------------------")
print(f"MAE:  {dummy_mae:.8f}")
print(f"RMSE: {dummy_rmse:.8f}")
print(f"R²:   {dummy_r2:.4f}")








