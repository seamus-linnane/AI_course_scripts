import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset and convert it into a DataFrame
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)  # Features
iris_df['target'] = iris.target  # Add the target column

# 'setosa' is mapped to 1, and all other species ('not-setosa') are mapped to 0
iris_df['binary_target'] = iris_df['target'].map({0: 1, 1: 0, 2: 0})  # Setosa = 1, Others = 0

# Split data into train and test sets
X = iris_df.iloc[:, :4].values  # Use all features
y = iris_df['binary_target'].values  # Binary target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perceptron Functions
def initialize_weights(n_features):
    weights = np.zeros(n_features)  # Initialize weights to 0
    bias = 0  # Initialize bias to 0
    return weights, bias

def activation_function(z):
    return 1 if z >= 0 else 0

def predict(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return activation_function(weighted_sum)

def train_with_tracking(X, y, learning_rate=0.1, epochs=10):
    weights, bias = initialize_weights(X.shape[1])
    training_history = {'epoch': [], 'total_error': [], 'weights': []}  # Dictionary to track progress

    for epoch in range(epochs):
        total_error = 0
        for inputs, target in zip(X, y):
            prediction = predict(inputs, weights, bias)
            error = target - prediction
            total_error += abs(error)  # Accumulate error for this epoch
            weights += learning_rate * error * inputs
            bias += learning_rate * error

        # Record training metrics
        training_history['epoch'].append(epoch + 1)
        training_history['total_error'].append(total_error)
        training_history['weights'].append(weights.copy())  # Copy weights to avoid overwriting
    
    return weights, bias, training_history

# Train the Perceptron with tracking
weights, bias, training_history = train_with_tracking(X_train, y_train, learning_rate=0.1, epochs=10)

# Test the Perceptron
y_pred = np.array([predict(sample, weights, bias) for sample in X_test])

# Evaluate the Results
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Iris-other", "Iris-setosa"]))

# Plot total error vs. epoch
plt.figure(figsize=(8, 6))
plt.scatter(training_history['epoch'], training_history['total_error'], color='blue')
plt.plot(training_history['epoch'], training_history['total_error'], color='blue', linestyle='--', label='Total Error')
plt.title("Perceptron Training: Total Error vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.legend()
plt.show()