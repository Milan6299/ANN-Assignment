from scipy.io import arff
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data, meta = arff.loadarff("vehicle_clean.arff")
df = pd.DataFrame(data)

# Convert class to 0/1
df["Class"] = (
    df["Class"].str.decode("utf-8").str.strip().map({"positive": 1, "negative": 0})
)

# Train-test split using Stratified Random Sampling
TR = df.iloc[:, :-1]
CL = df["Class"]
TR_train, TR_test, CL_train, CL_test = train_test_split(
    TR, CL, test_size=0.25, shuffle=True, stratify=CL, random_state=42
)

# Ensure TR_train / TR_test are DataFrames
TR_train = pd.DataFrame(TR_train)
TR_test = pd.DataFrame(TR_test)

# Ensure CL_train / CL_test are Series
CL_test = pd.Series(CL_test)
CL_train = pd.Series(CL_train)

# Z-score normalization
mean = TR_train.mean()
std = TR_train.std()
TR_train_norm = (TR_train - mean) / std
TR_test_norm = (TR_test - mean) / std

# Class counts for weights
n_total = len(CL_train)
class0_count = (CL_train == 0).sum()
class1_count = (CL_train == 1).sum()

weight0 = n_total / (2 * class0_count)  # negative class
weight1 = n_total / (2 * class1_count)  # positive class
class_weights = {0: weight0, 1: weight1}
print("Class Weights:", class_weights)

# ANN Parameters
input_size = TR_train.shape[1]
hidden_neurons = 4
alpha = 0.1
epochs = 700


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def xav_uniform(rows, columns):
    return np.sqrt(6 / (rows + columns))


# Weight initialization
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

# Prepare training data
X_train = TR_train_norm.values
Y_train = CL_train.to_numpy().reshape(-1, 1)

# Training loop with class weights
E = np.zeros((epochs, 1))
for e in range(epochs):
    total_err = 0
    for i in range(len(X_train)):
        Xi = X_train[i].reshape(1, -1)
        Yi_true = Y_train[i].reshape(1, 1)

        # Forward pass
        Z = Xi @ W + b_hidden
        A = relu(Z)
        Yin = A @ V + b_output
        Y_pred = sigmoid(Yin)

        # Determine class weight
        cw = class_weights[int(Yi_true[0, 0])]

        # Error
        err = Yi_true - Y_pred
        total_err += cw * (err**2)

        # Backpropagation
        dYin = cw * err * Y_pred * (1 - Y_pred) * (-1)
        dV = A.T @ dYin
        db_output = dYin
        dZ = (dYin @ V.T) * relu_derivative(Z)
        dW = Xi.T @ dZ
        db_hidden = dZ

        # Update weights
        W -= alpha * dW
        b_hidden -= alpha * db_hidden
        V -= alpha * dV
        b_output -= alpha * db_output

    E[e] = total_err / len(X_train)

# Plot training error
plt.plot(np.arange(1, epochs + 1), E.flatten(), color="blue")
plt.xlabel("Epoch")
plt.ylabel("Weighted MSE")
plt.title("Training Error vs Epoch")
plt.grid(True)
plt.show()


# Prediction function
def predict(x_row, W, b_hidden, V, b_output):
    Z = x_row @ W + b_hidden
    A = relu(Z)
    Yin = A @ V + b_output
    Y = sigmoid(Yin)
    return 1 if Y >= 0.5 else 0


# Test evaluation
def testing(TE, CTE):
    length = len(TE)
    correct = 0
    for i in range(length):
        x_row = TE.iloc[i].values.reshape(1, -1)
        actual = int(CTE.iloc[i])
        pred = predict(x_row, W, b_hidden, V, b_output)
        if pred == actual:
            correct += 1

    accuracy = correct / length
    print("Test Accuracy:", accuracy)


# Change the values of TE and CTE to check for different test sets
TE = df.iloc[:, :-1]
TE = (TE - mean) / std
CTE = df.iloc[:, -1]
testing(TE, CTE)
testing(TR_test_norm, CL_test)
