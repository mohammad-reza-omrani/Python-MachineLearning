import numpy as np

def relu(x):
    """
    Apply the ReLU activation function element-wise.
    
    Args:
    - x: The input array (can be a NumPy array, list, or scalar).
    
    Returns:
    - The element-wise ReLU activation (NumPy array or scalar).
    """
    return np.maximum(0, x)

# Example usage
if __name__ == "__main__":
    # Example input (NumPy array)
    x = np.array([-1, 0, 1, 2, -3, 4])
    
    # Apply ReLU function
    result = relu(x)
    
    print("ReLU Output:", result)
