import numpy as np
from scipy.stats import norm

x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, loc = 0, scale = 1)
samples = norm.rvs(loc = 0, scale = 1, size = 1000)
