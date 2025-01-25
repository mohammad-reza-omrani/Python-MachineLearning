import numpy as np
from scipy.stats import ttest_1samp

data = np.array([3, 5, 7, 8, 9, 10, 11, 14, 15, 17])
t_statistic, p_value = ttest_1samp(data, 10)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)
