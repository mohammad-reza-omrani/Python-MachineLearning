from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate synthetic data (features and labels)
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with a radial basis function (RBF) kernel
svm_classifier = SVC(kernel='rbf')

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = svm_classifier.predict(X_test)

# Print predicted labels
print("Predicted labels:", y_pred)

# Optional: Check accuracy (for example)
print("Accuracy:", accuracy_score(y_test, y_pred))
