import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

# Load data
data = pd.read_csv('traindata.txt', header=None)
labels = pd.read_csv('trainlabels.txt', header=None)

# Split data into training and validation to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000, random_state=42)  # Increased max_iter for convergence
model.fit(X_train, y_train.values.ravel())

# Predict on the test data
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the Logistic Regression model:", accuracy)

# Display the confusion matrix and classification report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Perform cross-validation
scores = cross_val_score(model, data, labels.values.ravel(), cv=5)  # 5-fold cross-validation
print("Average cross-validation score: %.2f" % scores.mean())

# Save the model
joblib.dump(model, 'logistic_regression_model.pkl')
