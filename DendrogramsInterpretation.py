import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data (100 samples, 2 features) for testing
X = np.random.randn(100, 2)

# Perform hierarchical clustering using a specific linkage criterion
Z = linkage(X, method='average')

# Generate a dendrogram
plt.figure(figsize=(10, 7))  # Optional: adjust the figure size for better clarity
dendrogram(Z)

# Set the distance threshold for the cut
distance_threshold = 50

# Draw a horizontal cut at the desired threshold
plt.axhline(y=distance_threshold, color='r', linestyle='--')

# Show the plot
plt.title('Dendrogram with Horizontal Cut')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()