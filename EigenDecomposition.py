import numpy as np

# Generate synthetic data (100 samples, 3 features)
X = np.random.randn(100, 3)

# Calculate the covariance matrix
cov_matrix = np.cov(X.T)

# Calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvals, eigenvects = np.linalg.eig(cov_matrix)

# Print eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvals)

print("\nEigenvectors:")
print(eigenvects)