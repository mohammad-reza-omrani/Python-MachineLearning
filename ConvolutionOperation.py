import numpy as np

def convolution(image, filter):
    """
    Perform 2D convolution of an image with a filter.

    Args:
    - image: The input image (2D numpy array).
    - filter: The filter or kernel to apply (2D numpy array).

    Returns:
    - The result of the convolution (2D numpy array).
    """
    height, width = image.shape
    f_height, f_width = filter.shape
    
    # Output dimensions (valid padding)
    output = np.zeros((height - f_height + 1, width - f_width + 1))

    # Perform convolution operation
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+f_height, j:j+f_width] * filter)

    return output

# Example usage
if __name__ == "__main__":
    # Example image (5x5) and filter (3x3)
    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])

    filter = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Perform convolution
    result = convolution(image, filter)
    print("Convolution Result:")
    print(result)
