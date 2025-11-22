from numpy import exp, size
from scipy.io import arff
import pandas as pd

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
print(ts)

hlayers = 2

def sigmoid(x):
    return 1/(1+ exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1-sx)

wlen = len(ts)
print(wlen)

print(sigmoid(800),sigmoid_derivative(800))

