import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the data
X = pd.read_csv('traindata.txt', header=None)  # Ensure this loads correctly as a DataFrame
y = pd.read_csv('trainlabels.txt', header=None)

# Flatten y to a one-dimensional array
y = y.squeeze()  # Converts DataFrame to Series if only one column

# Split data into training and validation sets to check performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model on the training data
gbm.fit(X_train, y_train)

# Predict on the validation set
y_pred = gbm.predict(X_val)

# Calculate and print the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2%}")

# Save the model to a file
joblib.dump(gbm, 'gradient_boosting_model.pkl')

