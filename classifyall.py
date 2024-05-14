import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('neural_network_model.h5')
scaler = joblib.load('scaler.pkl')

# Load test data
test_data = pd.read_csv('traindata.txt', header=None)

# Preprocess the test data using the loaded scaler
test_data_scaled = scaler.transform(test_data)

# Predict using the model
predictions = model.predict(test_data_scaled)
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Save predictions to predlabels.txt
np.savetxt('predlabels.txt', predicted_labels, fmt='%d')
