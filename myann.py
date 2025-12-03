from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")  # To make sure linux uses GUI

data, meta = arff.loadarff("vehicle_clean.arff")
df = pd.DataFrame(data)
print(df.head())

# Convert class from bytes to string and map to 0/1
df["Class"] = (
    df["Class"].str.decode("utf-8").str.strip().map({"positive": 1, "negative": 0})
)
# Count the number of class with value 0 and 1
dfsize = len(df)
class0 = (df["Class"] == 0).sum()
class1 = dfsize - class0
print("Class 0:", class0, "Class 1:", class1, "Total: ", dfsize)

# Intitializing training and test set
TR = df.iloc[:, :-1]
CL = df["Class"]

# Z-Score Normalization - in order to make the mean 0 and std 1
TR = (TR - TR.mean()) / TR.std()
print("TR", TR)
# Used stratified random sampling to split data keeping the ratio of class 0 to 1 constant
TR_train, TR_test, CL_train, CL_test = train_test_split(
    TR, CL, test_size=0.25, shuffle=True, stratify=CL, random_state=42
)

TRCL = pd.concat(
    [TR_train.reset_index(drop=True), CL_train.reset_index(drop=True)], axis=1
)
print("TRCL", TRCL)
trclsize = len(TRCL)
class0 = (TRCL["Class"] == 0).sum()
class1 = trclsize - class0
print("Class 0:", class0, "Class 1:", class1, "Total: ", trclsize)
# Handling Imbalanced Dataset
print((TRCL["Class"] == 1).sum())

# Separate majority and minority classes
TR_majority = TRCL[TRCL["Class"] == 0]
TR_minority = TRCL[TRCL["Class"] == 1]

# Downsample majority class to match minority class size
TR_majority_downsampled = resample(
    TR_majority,
    replace=False,  # sample without replacement
    n_samples=len(TR_minority),  # match minority class
    random_state=42,
)
# Combine minority class with downsampled majority class
TRCL_balanced = pd.concat([TR_minority, TR_majority_downsampled])

# Shuffle the dataset
TRCL_balanced = TRCL_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check counts after undersampling
print(TRCL_balanced["Class"].value_counts())
TRB = TRCL_balanced.iloc[:, :-1]
Ycap = TRCL_balanced.iloc[:, -1].values.reshape(-1, 1)
print(TRB)
print(Ycap)

input_size = TRB.shape[1]
hidden_neurons = 4
alpha = 0.1
epochs = 1000


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Xavier Initialization
def xav_uniform(rows, columns):
    return np.sqrt(6 / (rows + columns))


# Intitializing weights and bias
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

# print(W)

# Initialization of Error Matrix
E = np.zeros((epochs, 1))

for e in range(1):
    total_err = 0

    for i in range(len(TRB)):
        # Extract sample and row by row (Stochastic Method)
        Xi = TRB.iloc[i].values.reshape(1, -1)  # shape (1, 18)
        Yci = Ycap[i].reshape(1, 1)  # shape (1, 1)

        # Forward Pass
        Z = Xi @ W + b_hidden  # shape(1,4)
        # print(i, Z)
        A = sigmoid(Z)
        Yin = A @ V + b_output
        Y = sigmoid(Yin)

        # Error for this row
        err = Yci - Y
        total_err += err**2

        # Backpropagation
        dYin = Y * (1 - Y) * err * (-1)  # output layer delta - shape(1,1)
        dV = A.T @ dYin  # shape (1,4)
        db_output = dYin  # shape (1,1)

        dZ = (dYin @ V.T) * (A * (1 - A))  # hidden layer delta - shape(4,1)
        dW = Xi.T @ dZ  # shape (18,4)
        db_hidden = dZ  # shape (4,1)

        # Weight Updates
        W -= alpha * dW  # shape (18,4)
        b_hidden += alpha * db_hidden  # shape (1,4)
        V -= alpha * dV  # shape (1,4)
        b_output += alpha * db_output  # shape (1,1)

    # MSE(Mean Squared Error) for each epoch
    E[e] = np.mean(total_err)
# Comparison of data before and after undersampling
# sns.countplot(x="Class", data=TRCL)
# plt.show()
# sns.countplot(x="Class", data=df)
# plt.show()
