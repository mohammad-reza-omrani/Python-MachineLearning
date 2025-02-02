import numpy as np
from sklearn.mixture import GaussianMixture

# Generate synthetic data (100 samples, 2 features)
X = np.random.randn(100, 2)

# Define the number of components (clusters) for the GMM
K = 3

# Initialize the GaussianMixture model
gmm = GaussianMixture(n_components=K, random_state=0)

# Fit the model once to initialize the parameters
gmm.fit(X)

# Specify the convergence threshold
tolerance = 1e-3

# Initialize the previous log-likelihood value
prev_log_likelihood = -np.inf

# Perform the EM algorithm manually with convergence check
while True:
    # Perform the E-step: calculate the responsibilities
    responsibilities = gmm.predict_proba(X)

    # Perform the M-step: update the model parameters manually
    
    # Update means
    gmm.means_ = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]
    
    # Update covariances
    diff = X[:, np.newaxis, :] - gmm.means_[np.newaxis, :, :]
    gmm.covariances_ = np.array([
        np.dot(responsibilities[:, k] * diff[:, k, :].T, diff[:, k, :]) / np.sum(responsibilities[:, k])
        for k in range(K)
    ])
    
    # Update weights
    gmm.weights_ = np.mean(responsibilities, axis=0)
    
    # Calculate the current log-likelihood value
    current_log_likelihood = gmm.score(X)

    # Check for convergence (if the change in log-likelihood is below tolerance)
    if current_log_likelihood - prev_log_likelihood < tolerance:
        break

    # Update the previous log-likelihood value
    prev_log_likelihood = current_log_likelihood

# Print final model parameters after convergence
print("Final Means:")
print(gmm.means_)

print("\nFinal Covariances:")
print(gmm.covariances_)

print("\nFinal Weights:")
print(gmm.weights_)