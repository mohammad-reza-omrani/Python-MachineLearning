from sklearn.mixture import GaussianMixture
import numpy as np

# Generate synthetic data (100 samples, 2 features)
X = np.random.randn(100, 2)

# Specify the number of components (clusters) for the Gaussian Mixture Model
K = 3

# Create a GaussianMixture instance and specify the desired number of components
gmm = GaussianMixture(n_components=K, random_state=0)

# Fit the GaussianMixture model to the data
gmm.fit(X)

# Retrieve the cluster assignments for each data point
cluster_labels = gmm.predict(X)

# Print the cluster labels for each data point
print("Cluster Labels:", cluster_labels)
