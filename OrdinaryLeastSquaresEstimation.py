import numpy as np
import statsmodels.api as sm

# Independent variables
X = np.array([[1, 3], [1, 5], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 14], [1, 15], [1, 17]])

# Dependent variable
y = np.array([4, 7, 5, 12, 11, 13, 19, 20, 21, 25])

# Fit the model
model = sm.OLS(y, X)
results = model.fit()

# Obtain the coefficients
coefficients = results.params

# Obtain the residuals
residuals = results.resid

# Obtain the R-squared statistic
R_squared = results.rsquared