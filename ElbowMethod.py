import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Generate synthetic data (100 samples, 2 features)
X = np.random.randn(100, 2)

# Define the maximum number of clusters to test
max_k = 10

# Calculate the sum of squared distances for different values of K
ssd = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)  # inertia_ is the sum of squared distances

# Plot the sum of squared distances against different values of K
plt.plot(range(1, max_k + 1), ssd, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()