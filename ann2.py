from scipy.io import arff
import pandas as pd
import numpy as np

# --- 1. Load and preprocess ---
data, meta = arff.loadarff("vehicle_clean.arff")
df = pd.DataFrame(data)

# Convert class from bytes → string and map to 0/1
df["Class"] = (
    df["Class"].str.decode("utf-8").str.strip().map({"positive": 1, "negative": 0})
)

# Ensure numeric and normalize features
X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
X = (X - X.mean()) / X.std()  # Standardization
Y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

# Split into train/test
X_train = X.iloc[:800].to_numpy()
Y_train = Y[:800]
X_test = X.iloc[800:].to_numpy()
Y_test = Y[800:]

# --- 2. Parameters ---
input_size = X_train.shape[1]
hidden_neurons = 10
learning_rate = 0.1
epochs = 10000

# Class weights (inverse frequency)
classes, counts = np.unique(Y_train, return_counts=True)
class_weights = {c: np.max(counts) / count for c, count in zip(classes, counts)}


# --- 3. Activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# --- 4. Xavier Initialization ---
def xav_uniform(fan_in, fan_out):
    return np.sqrt(6 / (fan_in + fan_out))


# --- 5. Initialize weights and biases ---
W = np.random.uniform(
    -xav_uniform(input_size, hidden_neurons),
    xav_uniform(input_size, hidden_neurons),
    (input_size, hidden_neurons),
)
b_hidden = np.zeros((1, hidden_neurons))

V = np.random.uniform(
    -xav_uniform(hidden_neurons, 1), xav_uniform(hidden_neurons, 1), (hidden_neurons, 1)
)
b_output = np.zeros((1, 1))

# --- 6. Training ---
for epoch in range(epochs):
    # Forward pass
    Z1 = X_train @ W + b_hidden  # (N_samples, hidden_neurons)
    A1 = sigmoid(Z1)  # Hidden activations
    Yin = A1 @ V + b_output  # Output sum
    Ycal = sigmoid(Yin)  # Output activation

    # Compute class-weighted error
    weight_vector = np.array([class_weights[y[0]] for y in Y_train]).reshape(-1, 1)
    error_output = weight_vector * (Ycal - Y_train) * (Ycal * (1 - Ycal))

    # Hidden layer deltas
    delta_hidden = (error_output @ V.T) * (A1 * (1 - A1))

    # Gradients
    grad_V = A1.T @ error_output
    grad_b_output = np.sum(error_output, axis=0, keepdims=True)

    grad_W = X_train.T @ delta_hidden
    grad_b_hidden = np.sum(delta_hidden, axis=0, keepdims=True)

    # Update weights and biases
    V -= learning_rate * grad_V
    b_output -= learning_rate * grad_b_output
    W -= learning_rate * grad_W
    b_hidden -= learning_rate * grad_b_hidden

    # Optional: print loss
    if epoch % 1000 == 0:
        loss = np.mean(weight_vector * np.square(Ycal - Y_train))
        print(f"Epoch {epoch}, Weighted MSE Loss: {loss:.4f}")

# --- 7. Testing ---
Z1_test = X_test @ W + b_hidden
A1_test = sigmoid(Z1_test)
Y_test_pred = sigmoid(A1_test @ V + b_output)
predictions = (Y_test_pred >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predictions == Y_test) * 100
print(f"\nTesting Accuracy: {accuracy:.2f}%")

# Side-by-side comparison of first 20 test samples
print("\nFirst 20 Test Samples Comparison:")
print(f"{'Index':<5} {'Predicted':<10} {'Actual':<10} {'Prob':<10} {'Features'}")
for i in range(min(20, X_test.shape[0])):
    features_str = ", ".join([f"{f:.2f}" for f in X_test[i]])
    print(
        f"{i:<5} {predictions[i][0]:<10} {Y_test[i][0]:<10} {Y_test_pred[i][0]:<10.4f} {features_str}"
    )
