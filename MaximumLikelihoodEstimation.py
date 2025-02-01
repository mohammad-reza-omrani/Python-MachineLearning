from sklearn.linear_model import LogisticRegression

# Independent variables
X = [[1, 3], [1, 5], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 14], [1, 15], [1, 17]]

# Dependent variable
y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]

# Fit the model
model = LogisticRegression()
model.fit(X, y)

# Obtain the coefficients
coefficients = model.coef_
intercept = model.intercept_