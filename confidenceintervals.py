import numpy as np
from scipy.stats import t

data = np.array([3, 5, 7, 8, 9, 10, 11, 14, 15, 17])

sample_mean = np.mean(data)
sample_std = np.std(data)
sample_size = len(data)

confidence_level = 0.95
degree_of_freedom = sample_size - 1

critical_value = t.ppf((1 + confidence_level) / 2, df=degree_of_freedom)
margin_of_error = critical_value * (sample_std / np.sqrt(sample_size))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("Confidence Interval:", confidence_interval)
