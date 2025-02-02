from scipy.spatial.distance import euclidean
import numpy as np

# Define two data points (e.g., 2-dimensional points)
x_i = np.array([1, 2])
x_j = np.array([4, 6])

# Calculate the Euclidean distance between x_i and x_j
distance = euclidean(x_i, x_j)

# Print the result
print("Euclidean Distance:", distance)