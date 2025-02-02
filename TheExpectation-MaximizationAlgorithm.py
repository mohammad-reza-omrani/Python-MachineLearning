import numpy as np
from sklearn.mixture import GaussianMixture

# Generate synthetic data (100 samples, 2 features)
X = np.random.randn(100, 2)

# Define the number of components (clusters) for the GMM
K = 3

# Initialize the GaussianMixture model
gmm = GaussianMixture(n_components=K, random_state=0)

# Fit the GMM to the data
gmm.fit(X)

# Perform the E-step: calculate the responsibilities
responsibilities = gmm.predict_proba(X)

# Perform the M-step: update the model parameters manually

# Update means
gmm.means_ = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]

# Update covariances
# Calculate the difference (X - means) for each component
diff = X[:, np.newaxis, :] - gmm.means_[np.newaxis, :, :]
# Compute the weighted sum of outer products and divide by the sum of responsibilities
gmm.covariances_ = np.array([
    np.dot(responsibilities[:, k] * diff[:, k, :].T, diff[:, k, :]) / np.sum(responsibilities[:, k])
    for k in range(K)
])

# Update weights
gmm.weights_ = np.mean(responsibilities, axis=0)

# Print updated parameters
print("Updated Means:")
print(gmm.means_)

print("\nUpdated Covariances:")
print(gmm.covariances_)

print("\nUpdated Weights:")
print(gmm.weights_)
