import numpy as np

# Initialize variables
learning_rate = 0.01
batch_size = 32
max_iterations = 1000
epsilon = 1e-6

# Initialize x
x = np.array([1.0, 1.0])

# Define the gradient function with proper indentation
def gradient(x, batch):
    return np.sum(2 * batch, axis=0)

# Define your training data (X), for example:
X = np.random.randn(100, 2)  # Random data with 100 samples and 2 features

# Perform mini-batch gradient descent
for iteration in range(max_iterations):
    
    # Randomly select mini-batch
    indices = np.random.choice(len(X), size=batch_size, replace=False)
    batch = X[indices]

    # Update x
    x_new = x - learning_rate * gradient(x, batch)

    # Check for convergence
    if np.linalg.norm(x_new - x) < epsilon:
        break

    x = x_new

# Print the final x
print(x)