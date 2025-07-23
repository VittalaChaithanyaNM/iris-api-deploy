import csv
import numpy as np

# Load the data from iris.csv
features = []
labels = []

with open('../iris.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    for row in reader:
        features.append([float(x) for x in row[:4]])
        labels.append(row[4])

# Convert to numpy arrays
X = np.array(features)
classes = sorted(list(set(labels)))
class_to_index = {label: i for i, label in enumerate(classes)}
y = np.array([class_to_index[label] for label in labels])

# One-hot encode y
Y = np.zeros((y.size, len(classes)))
Y[np.arange(y.size), y] = 1

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize weights randomly
np.random.seed(42)
weights = np.random.randn(X_bias.shape[1], len(classes)) * 0.01

# Hyperparameters
lr = 0.1
epochs = 500

# Training using basic gradient descent
for epoch in range(epochs):
    logits = X_bias.dot(weights)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    loss = -np.sum(Y * np.log(probs + 1e-8)) / X.shape[0]

    grad = X_bias.T.dot(probs - Y) / X.shape[0]
    weights -= lr * grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Separate weights and bias for saving
biases = weights[0, :]
weights_only = weights[1:, :]

# Save as .npy files
np.save("weights.npy", weights_only)
np.save("biases.npy", biases)

print("Training complete. Weights and biases saved.")
