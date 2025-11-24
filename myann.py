from scipy.io import arff
import pandas as pd
import numpy as np

data, meta = arff.loadarff("vehicle_clean.arff")
df = pd.DataFrame(data)

# Convert class from bytes → string and map to 0/1
df["Class"] = (
    df["Class"].str.decode("utf-8").str.strip().map({"positive": 1, "negative": 0})
)
# Count the number of class with value 0 and 1
class0 = (df["Class"] == 0).sum()
class1 = len(df) - class0
print("Class 0:", class0, "Class 1:", class1, "Total: ", len(df))

# Handling Imbalanced Dataset


# Intitializing training and test set
TR = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
print(TR.head())
