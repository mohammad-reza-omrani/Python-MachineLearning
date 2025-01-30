import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Independent variable
X = np.array([3, 5, 7, 8, 9, 10, 11, 14, 15, 17]).reshape(-1, 1)

# Dependent variable
y = np.array([4, 7, 5, 12, 11, 13, 19, 20, 21, 25])

# Create linear regression object
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict the dependent variable
y_pred = model.predict(X)

# Calculate RSS, TSS, and R^2
RSS = mean_squared_error(y, y_pred) * len(y)
TSS = mean_squared_error(y, np.full_like(y, np.mean(y))) * len(y)
R_squared = r2_score(y, y_pred)