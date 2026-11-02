
import numpy as np
from scipy.io import loadmat

# Load only the matrices required for machine learning
data = loadmat(
    "DTMB5415-database.mat",
    variable_names=["variablesMatrix", "coefStatMatrix"]
)


geometry = data["variablesMatrix"]
resistance = data["coefStatMatrix"]


# Transpose: rows become hull configurations
geometry = geometry.T
resistance = resistance.T


# Operating conditions
froude_numbers = np.array([0.25, 0.33, 0.41])


# Repeat each hull's geometry for its three operating speeds
X_geometry = np.repeat(geometry, 3, axis=0)


# Assign the three Froude numbers to each hull
X_froude = np.tile(froude_numbers, len(geometry))


# Combine geometry and operating speed into 28 features
X = np.column_stack((X_geometry, X_froude))


# Flatten resistance values to match the input rows
y = resistance.flatten()


# Track which hull produced each observation
hull_ids = np.repeat(np.arange(1, len(geometry) + 1), 3)

print("Input matrix:", X.shape)
print("Target array:", y.shape)
print("Hull IDs:", hull_ids.shape)

print("\nFirst three observations:")
for i in range(3):
    print(
        f"Hull: {hull_ids[i]}, "
        f"Fr: {X[i, -1]}, "
        f"Cw: {y[i]:.8f}"
    )






## DATA SPLITTING




from sklearn.model_selection import GroupShuffleSplit

# First split: 70% training and 30% temporary data
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(X, y, groups=hull_ids)
)



# Second split: divide temporary data equally
# into validation and test sets
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative, test_relative = next(
    split2.split(
        X[temp_idx],
        y[temp_idx],
        groups=hull_ids[temp_idx]
    )
)

# Convert temporary indices back to original positions
val_idx = temp_idx[val_relative]
test_idx = temp_idx[test_relative]

# Create the final datasets
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Check the resulting dimensions
print("\nDataset splits:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

# Verify no hull appears in more than one dataset
train_hulls = set(hull_ids[train_idx])
val_hulls = set(hull_ids[val_idx])
test_hulls = set(hull_ids[test_idx])

assert train_hulls.isdisjoint(val_hulls)
assert train_hulls.isdisjoint(test_hulls)
assert val_hulls.isdisjoint(test_hulls)

print("\nHull overlap check: PASSED")
print("Training hulls:", len(train_hulls))
print("Validation hulls:", len(val_hulls))
print("Testing hulls:", len(test_hulls))







## FEATURE SCALING AND STANDARDISATION




from sklearn.preprocessing import StandardScaler

# Create a feature scaler
scaler = StandardScaler()


# Learn scaling parameters using ONLY training data
X_train_scaled = scaler.fit_transform(X_train)


# Apply the same transformation to validation and testing
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Inspect the scaling parameters
print("\nFeature scaling:")


print("Training feature means:")
print(np.round(scaler.mean_, 4))


print("\nTraining feature standard deviations:")
print(np.round(scaler.scale_, 4))


print("\nFirst training observation (before scaling):")
print(np.round(X_train[0], 4))


print("\nFirst training observation (after scaling):")
print(np.round(X_train_scaled[0], 4))


# Confirm shapes remain unchanged
print("\nScaled dataset shapes:")
print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Testing:", X_test_scaled.shape)








## SAVING PROCESSED DATA



from pathlib import Path
import joblib

# Create a folder for processed data
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# Save the numerical datasets
np.savez_compressed(
    output_dir / "dataset.npz",

    X_train=X_train_scaled,
    X_val=X_val_scaled,
    X_test=X_test_scaled,

    y_train=y_train,
    y_val=y_val,
    y_test=y_test,

    train_hull_ids=hull_ids[train_idx],
    val_hull_ids=hull_ids[val_idx],
    test_hull_ids=hull_ids[test_idx],

    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

# Save the fitted feature scaler
joblib.dump(
    scaler,
    output_dir / "feature_scaler.joblib"
)

print("\nProcessed data saved successfully!")
print("Location:", output_dir.resolve())
