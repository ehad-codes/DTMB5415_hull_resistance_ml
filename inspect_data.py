
from scipy.io import loadmat

file_path = "DTMB5415-database.mat"

# Load only the hull design parameters and wave resistance data
data = loadmat(
    file_path,
    variable_names=["variablesMatrix", "coefStatMatrix"]
)

print("Variables loaded:")
print(data.keys())

# Extract the two numerical arrays
X = data["variablesMatrix"]
Y = data["coefStatMatrix"]

print("\nHull geometry parameters:", X.shape)
print("Wave resistance coefficients:", Y.shape)

# Inspect the baseline configuration
print("\nBaseline hull design parameters:")
print(X[:, 0])

print("\nBaseline wave resistance coefficients:")
print(Y[:, 0])




import numpy as np

# Inspect the range of the geometry parameters
print("\nGeometry parameters:")
print("Minimum:", np.min(X))
print("Maximum:", np.max(X))




# Inspect wave resistance for each operating condition
froude_numbers = [0.25, 0.33, 0.41]

for i, fr in enumerate(froude_numbers):
    resistance = Y[i, :]

    print(f"\nFroude number: {fr}")
    print("Minimum Cw:", np.min(resistance))
    print("Maximum Cw:", np.max(resistance))
    print("Mean Cw:", np.mean(resistance))
    print("Negative values:", np.sum(resistance < 0))




# Find hull configurations with negative Cw at Fr = 0.41
negative_indices = np.where(Y[2, :] < 0)[0]

print("Hull indices with negative Cw:", negative_indices)

for index in negative_indices:
    print("Hull number (MATLAB indexing):", index + 1)
    print("Wave resistance coefficient:", Y[2, index])
    print("Design variables:", X[:, index])












# Investigate hull 1937
hull_index = 1936  # Python uses zero-based indexing

print("\nHull 1937 resistance across all speeds:")

for i, fr in enumerate([0.25, 0.33, 0.41]):
    cw = Y[i, hull_index]
    print(f"Fr = {fr}: Cw = {cw:.8f}")

# Identify the configuration with maximum Cw at Fr = 0.25
max_index = np.argmax(Y[0, :])

print("\nHighest resistance at Fr = 0.25:")
print("Hull number:", max_index + 1)
print("Cw:", Y[0, max_index])

print("\nIts resistance across all speeds:")
print(Y[:, max_index])
