import numpy as np

arrayOne = np.array([[1, 2], [3, 4]])
arrayTwo = np.array([[5, 6], [7, 8]])

resultForMatmul = np.matmul(arrayOne, arrayTwo)         # Matrix multiplication
resultForAdsign = arrayOne @ arrayTwo                   # Matrix multiplication using @ operator

print("Matrix Multiplication With matmul:")
print(resultForMatmul)
print("Matrix Multiplication With resultForAdsign:")
print(resultForAdsign)
