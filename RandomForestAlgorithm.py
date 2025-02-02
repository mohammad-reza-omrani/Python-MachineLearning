from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate synthetic data (features and labels)
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = rf_classifier.predict(X_test)

# Print predicted labels
print("Predicted labels:", y_pred)

# Optional: Check accuracy (for example)
print("Accuracy:", accuracy_score(y_test, y_pred))
