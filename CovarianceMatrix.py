import numpy as np

# Generate some synthetic data (100 samples, 3 features)
X = np.random.randn(100, 3)

# Calculate the covariance matrix (of features)
cov_matrix = np.cov(X.T)  # X.T is the transpose of X to get features as rows

# Print the covariance matrix
print("Covariance Matrix:")
print(cov_matrix)