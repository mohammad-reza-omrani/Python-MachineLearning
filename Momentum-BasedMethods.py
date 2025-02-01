import numpy as np

# Initialize variables
learning_rate = 0.1
momentum = 0.9
max_iterations = 1000
epsilon = 1e-6

# Initialize x and v
x = np.array([1.0, 1.0])
v = np.zeros_like(x)

# Define the gradient function with proper indentation
def gradient(x):
    return np.array([2 * x[0], 4 * x[1]])

# Perform momentum-based gradient descent
for iteration in range(max_iterations):
    # Compute gradient
    grad = gradient(x)

    # Update v and x
    v = momentum * v + learning_rate * grad
    x -= v

    # Check for convergence (using gradient norm for convergence)
    if np.linalg.norm(grad) < epsilon:  # Checking gradient norm for convergence
        break

# Output the final x
print(x)
