import numpy as np

arrayOne = np.array([[1, 2], [3, 4]])
eigenValues, eigenVectors = np.linalg.eig(arrayOne)

print("EigenValues:")
print(eigenValues)

print("EigenVectors:")
print(eigenVectors)
