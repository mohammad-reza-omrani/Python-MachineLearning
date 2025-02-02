from sklearn.decomposition import PCA
import numpy as np

# Generate synthetic data (100 samples, 5 features)
X = np.random.randn(100, 5)

# Specify the number of components for PCA (e.g., 2 components)
k = 2

# Create a PCA instance and specify the desired number of components
pca = PCA(n_components=k)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the lower-dimensional space
X_reduced = pca.transform(X)

# Print the reduced data (lower-dimensional representation)
print("Reduced Data:")
print(X_reduced)