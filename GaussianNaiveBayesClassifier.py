from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
import numpy as np

# Generate synthetic data (features and labels)
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = clf.predict(X_test)

# Print predicted labels
print("Predicted labels:", y_pred)

# Optional: You can also check accuracy (for example)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))