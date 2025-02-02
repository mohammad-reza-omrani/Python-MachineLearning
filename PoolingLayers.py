import numpy as np

def max_pooling(feature_map, stride):
    """
    Perform max pooling operation on the feature map.

    Args:
    - feature_map: The input feature map (2D numpy array).
    - stride: The stride length (integer).

    Returns:
    - The result of the max pooling operation (2D numpy array).
    """
    height, width = feature_map.shape
    pool_height = height // stride
    pool_width = width // stride
    output = np.zeros((pool_height, pool_width))

    # Perform max pooling
    for i in range(pool_height):
        for j in range(pool_width):
            output[i, j] = np.max(feature_map[i*stride:(i+1)*stride, j*stride:(j+1)*stride])

    return output

# Example usage
if __name__ == "__main__":
    # Example feature map (6x6)
    feature_map = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35, 36]
    ])

    stride = 2  # Pooling stride

    # Perform max pooling
    pooled_output = max_pooling(feature_map, stride)
    print("Pooled Output:")
    print(pooled_output)
