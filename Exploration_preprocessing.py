import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Load data
df = pd.read_csv('Machinelearning/iris.csv', header = 0)

#### Some data exploration
# data size
print("DATA EXPLORATION")
print("Data sample:")
print(df[0:5])

print("\nFeatures:")
for c in df.columns:
    print("\t"+c)

print("\nNumber of samples: "+str(df.shape[0]))

input("\nPress Enter to continue...")

# distibution of a feature
plt.hist(df['sepal_length'],8)
plt.xlabel('sepal_length')
plt.ylabel('frequency')
plt.show()

input("\nPress Enter to continue...")

# correlations
print("\nCORRELATIONS:")
print(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr())

input("\nPress Enter to continue...")

#### Scaling: z-score normalization
print("\nSCALING")
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:4])

print("\nMeans of original data:")
print(df.mean())
print("\nStandard deviations of original data:")
print(df.std())

print("\nMeans of transformed data:")
print(df_scaled.mean())
print("\nStandard deviations of transformed data:")
print(df_scaled.std())