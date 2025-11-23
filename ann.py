from numpy import exp, sqrt
from scipy.io import arff
import pandas as pd
import numpy as np

data, meta = arff.loadarff("vehicle_clean.arff")

df = pd.DataFrame(data)

# Convert class from bytes → string
if 'Class' in df.columns:
    df['Class'] = df['Class'].str.decode('utf-8')


df["Class"] = df["Class"].str.strip().map({
    "positive": 1,
    "negative": 0
})
ts = df.iloc[0:800]
# print(ts)

neurons = 2

def sigmoid(x):
    return 1/(1+ exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1-sx)

def xav_uniform(x):
    xav_sum = x + neurons
    return sqrt(6/xav_sum)

tsinputsize = len(ts.columns)-1
limit = xav_uniform(tsinputsize)

W = np.random.uniform(-limit, limit, (tsinputsize,neurons))
V = np.random.uniform(-limit, limit, (neurons))
Ycap = ts.iloc[:,tsinputsize].to_numpy()
print(Ycap)
delta = []
rowdata = ts.iloc[0,0:tsinputsize].to_numpy()

print(rowdata)

def calc_z(x):
    z = []  # store neuron sums
    for i in range(neurons):  # for each neuron
        sum_val = 0
        for j in range(tsinputsize):  # for each input
            sum_val += x[j] * W[j, i]
        z.append(sum_val)  # append sum for this neuron
    return z

print(calc_z(rowdata))
#Training Phase
def mlpbp(x):
    data = x.iloc[:, 0:tsinputsize].to_numpy()
    lr = 0.01  # learning rate
    for epoch in range(1000):
        for i in range(data.shape[0]):  # loop over all samples
            rowdata = data[i, :]  # current sample

            #Forward pass: calculate Z and A
            Z = []
            A = []
            for n in range(neurons):
                sum_val = 0
                for k in range(tsinputsize):
                    sum_val += rowdata[k] * W[k, n]
                Z.append(sum_val)
                A.append(sigmoid(sum_val))

            #Forward pass: calculate Yin and Ycal
            Yin = 0
            for n in range(neurons):
                Yin += A[n] * V[n]
            Ycal = sigmoid(Yin)

            #Calculate output error (sigma)
            sigma = (Ycal - Ycap[i]) * Ycal * (1 - Ycal)

            #Calculate hidden layer deltas
            delta = []
            for n in range(neurons):
                delta.append(A[n] * (1 - A[n]) * V[n])

            #Update output weights V
            for n in range(neurons):
                V[n] -= lr * sigma * A[n]

            #Update Weights in the input to hidden layer

            for n in range(neurons):
                for k in range(tsinputsize):
                    W[k, n] -= lr * delta[n] * sigma * rowdata[k]
