import numpy as np
import matplotlib.pyplot as plt

data = np.array([3, 5, 7, 8, 9, 10, 11, 14, 15, 17])

mean_value = np.mean(data)
median_value = np.median(data)
std_dev = np.std(data)

plt.boxplot(data)
plt.show()
