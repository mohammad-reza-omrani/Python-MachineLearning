import numpy as np

# Initialize variables
learning_rate = 0.1
max_iterations = 1000
epsilon = 1e-6

# Initialize x
x = np.array([1.0, 1.0])
G = np.zeros_like(x)

# Define the gradient function with proper indentation
def gradient(x):
    return np.array([2 * x[0], 4 * x[1]])

# Perform AdaGrad
for iteration in range(max_iterations):
    # Compute gradient
    grad = gradient(x)

    # Update G and x
    G += grad**2
    x -= (learning_rate / np.sqrt(G + epsilon)) * grad

    # Check for convergence (convergence is typically based on gradient norm or x change)
    if np.linalg.norm(grad) < epsilon:  # Checking gradient norm for convergence
        break

# Output the final x
print(x)
