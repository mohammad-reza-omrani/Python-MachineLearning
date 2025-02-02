from sklearn.cluster import KMeans
import numpy as np

# Generate synthetic data (100 samples, 2 features)
X = np.random.randn(100, 2)

# Specify the number of clusters (for example, 3 clusters)
K = 3

# Create a KMeans instance and specify the desired number of clusters
kmeans = KMeans(n_clusters=K, random_state=0)

# Fit the KMeans model to the data
kmeans.fit(X)

# Retrieve the cluster assignments for each data point
cluster_labels = kmeans.labels_

# Print the cluster labels for each data point
print("Cluster Labels:", cluster_labels)