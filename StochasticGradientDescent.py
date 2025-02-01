import numpy as np

# Initialize variables
learning_rate = 0.01
max_iterations = 1000
epsilon = 1e-6

# Initialize x
x = np.array([1.0, 1.0])

# Define the gradient function with proper indentation
def gradient(x):
    return np.array([2 * x[0], 4 * x[1]])

# Assuming X is your training data. Define it here, for example:
X = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])  # Example data points

# Perform stochastic gradient descent
for iteration in range(max_iterations):
    # Randomly select training example
    t = np.random.randint(0, len(X))
    xi = X[t]

    # Update x
    x_new = x - learning_rate * gradient(xi)

    # Check for convergence
    if np.linalg.norm(x_new - x) < epsilon:
        break

    x = x_new

print(x)
