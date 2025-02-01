import numpy as np

# Define the predictor variables X and the response variable y
# Example data (you can replace these with your own data)
X = np.array([[1], [2], [3], [4], [5]])  # Predictor variable (feature)
y = np.array([1, 2, 3, 4, 5])  # Response variable (target)

# Add a column of ones to X for the intercept term
X = np.column_stack((np.ones(len(X)), X))

# Compute the OLS estimator (beta_hat)
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Print the estimated coefficients (intercept and slope)
print("Estimated coefficients (intercept and slope):", beta_hat)
