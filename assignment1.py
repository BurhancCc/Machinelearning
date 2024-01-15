import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#### Load data
df = pd.read_csv(r'MachineLearning\FIFA-2019.csv')

#### Some data exploration
print("Data sample:")
print(df[0: 20])

print("\nNumber of samples: "+str(df.shape[0]))

correlation = df.corr()
print(correlation.shape)

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)

plt.show()

"""
for c in df.columns:
    print(c)
    plt.hist(df[c],10)
    plt.xlabel(c)
    plt.ylabel('frequency')
    plt.show()
"""

#### Scaling: z-score normalization
print("\nSCALING")
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Workrate_Low/ High', 'Workrate_Low/ Medium', 'Workrate_High/ Low', 'Workrate_Medium/ Low',
        'Workrate_Medium/ Medium', 'Workrate_High/ High', 'Workrate_High/ Medium', 'Workrate_Medium/ High', 'Workrate_Low/ Low', 'Body_Type_Lean', 'Body_Type_Normal', 'Body_Type_Stocky', 'Height', 'Weight', 'Crossing', 'Finishing',
        'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',	'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
        'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:58])

print("\nMeans of original data:")
print(df.mean())
print("\nStandard deviations of original data:")
print(df.std())

print("\nMeans of transformed data:")
print(df_scaled.mean())
print("\nStandard deviations of transformed data:")
print(df_scaled.std())

# split data
X = df[['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Workrate_Low/ High', 'Workrate_Low/ Medium', 'Workrate_High/ Low', 'Workrate_Medium/ Low',
        'Workrate_Medium/ Medium', 'Workrate_High/ High', 'Workrate_High/ Medium', 'Workrate_Medium/ High', 'Workrate_Low/ Low', 'Body_Type_Lean', 'Body_Type_Normal', 'Body_Type_Stocky', 'Height', 'Weight', 'Crossing', 'Finishing',
        'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',	'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
        'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y = df['Release Clause']

# build a k-NN model with k=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

# use the model to predict new example and should predict 0.299298861
predicted = clf.predict([[0.533333333,0.84,0.89,0.298319328,0.157894737,0.782916667,1,0.5,0.5,0.25,0,0,0,0,1,0,0,0,0,1,0,0,0.847058824,0.66,0.72,0.33,0.8,0.81,0.31,0.59,0.49,0.32,0.78,0.76,0.73,0.76,0.69,0.81,0.68,0.53,0.86,0.75,0.74,0.35,0.8,0.84,0.46,0.51,0.31,0.79,0.85,0.87,0.87,0.06,0.06,0.12,0.1,0.13]])
print(predicted)

# build a Naïve Bayes model
clfNaiveBayes = GaussianNB()
clfNaiveBayes.fit(X, y)

# use the model to predict new example and should predict 0.299298861
predictedNaiveBayes = clfNaiveBayes.predict([[0.533333333,0.84,0.89,0.298319328,0.157894737,0.782916667,1,0.5,0.5,0.25,0,0,0,0,1,0,0,0,0,1,0,0,0.847058824,0.66,0.72,0.33,0.8,0.81,0.31,0.59,0.49,0.32,0.78,0.76,0.73,0.76,0.69,0.81,0.68,0.53,0.86,0.75,0.74,0.35,0.8,0.84,0.46,0.51,0.31,0.79,0.85,0.87,0.87,0.06,0.06,0.12,0.1,0.13]])
print(predictedNaiveBayes)

