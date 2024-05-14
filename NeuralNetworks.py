import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load data
data = pd.read_csv('traindata.txt', header=None)
labels = pd.read_csv('trainlabels.txt', header=None)

# Data preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert labels to categorical since we are dealing with multi-class classification
labels = tf.keras.utils.to_categorical(labels, num_classes=21)  # Adjust num_classes if different

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(64, activation='relu'),                                   # Hidden layer
    Dense(21, activation='softmax')                                  # Output layer, adjust the number of units according to the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save the model and scaler
model.save('neural_network_model.h5')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use in preprocessing new data
