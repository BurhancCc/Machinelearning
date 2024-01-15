from audioop import cross
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, neighbors
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#### Load data
df = pd.read_csv('MachineLearning\FIFA-2019.csv')

# Dropping the columns that are not important
df = df.drop(columns = ["ID", "Name", "Photo", "Nationality", "Flag", "Club", "Club Logo", "Real Face", "Position", "Jersey Number", "Joined", "Loaned From", "Contract Valid Until",
              "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
              "CB", "RCB", "RB"])

# Cleaning up the relevant columns and dropping the rows with NaN values
df.drop(df[df['Body Type'] == "C. Ronaldo"].index, inplace = True)
df.drop(df[df['Body Type'] == "Akinfenwa"].index, inplace = True)
df.drop(df[df['Body Type'] == "Courtois"].index, inplace = True)
df.drop(df[df['Body Type'] == "Messi"].index, inplace = True)
df.drop(df[df['Body Type'] == "Neymar"].index, inplace = True)
df.drop(df[df['Body Type'] == "PLAYER_BODY_TYPE_25"].index, inplace = True)
df.drop(df[df['Body Type'] == "Shaqiri"].index, inplace = True)
df = df.dropna()
df["Value"] = df["Value"].str.replace("€", "")
df["Wage"] = df["Wage"].str.replace("€", "")
df["Release Clause"] = df["Release Clause"].str.replace("€", "")
df["Preferred Foot"] = df["Preferred Foot"].str.replace("Right", "1")
df["Preferred Foot"] = df["Preferred Foot"].str.replace("Left", "0")
df["Weight"] = df["Weight"].str.replace("lbs", "")

# One-Hot encoding for Body Type and dropping the old Body Type column
y = pd.get_dummies(df["Body Type"], prefix='Body_Type')
df["Body_Type_Lean"] = y["Body_Type_Lean"]
df["Body_Type_Normal"] = y["Body_Type_Normal"]
df["Body_Type_Stocky"] = y["Body_Type_Stocky"]
df = df.drop(columns = ["Body Type"])

# One-Hot encoding for Work Rate and dropping the old Work Rate Column
y = pd.get_dummies(df["Work Rate"], prefix='Work_Rate')
df["Work_Rate_High/ High"] = y["Work_Rate_High/ High"]
df["Work_Rate_High/ Medium"] = y["Work_Rate_High/ Medium"]
df["Work_Rate_High/ Low"] = y["Work_Rate_High/ Low"]
df["Work_Rate_Medium/ High"] = y["Work_Rate_Medium/ High"]
df["Work_Rate_Medium/ Medium"] = y["Work_Rate_Medium/ Medium"]
df["Work_Rate_Medium/ Low"] = y["Work_Rate_Medium/ Low"]
df["Work_Rate_Low/ High"] = y["Work_Rate_Low/ High"]
df["Work_Rate_Low/ Medium"] = y["Work_Rate_Low/ Medium"]
df["Work_Rate_Low/ Low"] = y["Work_Rate_Low/ Low"]
df = df.drop(columns = ["Work Rate"])

# Rewriting the Value, Wage, Release Clause and Height column to something we can use
for ind in df.index:
    valueRow = df["Value"][ind]
    if "M" in valueRow:
        valueRow = valueRow.replace("M", "")
        newR = float(valueRow)
        newR = newR * 1000000
        df["Value"][ind] = newR

    if "K" in valueRow:
        valueRow = valueRow.replace("K", "")
        newR = float(valueRow)
        newR = newR * 1000
        df["Value"][ind] = newR

    wageRow = df["Wage"][ind]
    if "M" in wageRow:
        wageRow = wageRow.replace("M", "")
        newR = float(wageRow)
        newR = newR * 1000000
        df["Wage"][ind] = newR

    if "K" in wageRow:
        wageRow = wageRow.replace("K", "")
        newR = float(wageRow)
        newR = newR * 1000
        df["Wage"][ind] = newR

    releaseClauseRow = df["Release Clause"][ind]
    if "M" in releaseClauseRow:
        releaseClauseRow = releaseClauseRow.replace("M", "")
        newR = float(releaseClauseRow)
        newR = newR * 1000000
        df["Release Clause"][ind] = newR

    if "K" in releaseClauseRow:
        releaseClauseRow = releaseClauseRow.replace("K", "")
        newR = float(releaseClauseRow)
        newR = newR * 1000
        df["Release Clause"][ind] = newR

    heightRow = df["Height"][ind]
    footAndInch = heightRow.split("'")
    inch = (int(footAndInch[0]) * 12) + int(footAndInch[1])
    df["Height"][ind] = inch

# Normalizing the dataset between 0 and 1
scaler = MinMaxScaler()
scaler.fit(df)
scaled = scaler.fit_transform(df)
df = pd.DataFrame(scaled, columns = df.columns)

# used to determine the bins to classifier
mean = df['Release Clause'].mean()
std = df['Release Clause'].std()

# the classifier variables here are the upper edge of the bins
# first bin in the range from 0  to mean/8
classifier1 = mean/8

# second bin in the range from mean/8 - mean/4
classifier2 = mean/4

# third bin in the range from mean/4 to mean 
classifier3 = mean 

# fourth bin in the range from mean to mean + std 
classifier4 = mean + std 

# fifth bin in the range from meant + std to mean + 2std
classifier5 = mean + std + std 

# sixt bin in the range from mean + 2std to 1
classifier6 = 1.01


# implementing the bins in the dataset
for ind in df.index:
    releaseClauseRow = df["Release Clause"][ind]

    if releaseClauseRow < classifier1:
        df["Release Clause"][ind] = "Really_Cheap"

    elif releaseClauseRow < classifier2:
        df["Release Clause"][ind] = "Fairly_Cheap"

    elif releaseClauseRow < classifier3:
        df["Release Clause"][ind] = "Below_Average"

    elif releaseClauseRow < classifier4:
        df["Release Clause"][ind] = "Above_Average"

    elif releaseClauseRow < classifier5:
        df["Release Clause"][ind] = "Fairly_Expensive"

    else:
        df["Release Clause"][ind] = "Really_Expensive"

print(df["Release Clause"].value_counts())

# split into train test sets
X = df[['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work_Rate_Low/ High', 'Work_Rate_Low/ Medium', 'Work_Rate_High/ Low', 'Work_Rate_Medium/ Low',
        'Work_Rate_Medium/ Medium', 'Work_Rate_High/ High', 'Work_Rate_High/ Medium', 'Work_Rate_Medium/ High', 'Work_Rate_Low/ Low', 'Body_Type_Lean', 'Body_Type_Normal', 'Body_Type_Stocky', 'Height', 'Weight', 'Crossing', 'Finishing',
        'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',	'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
        'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y = df['Release Clause']

# creating our train, and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)

def findOptimalSVMParameters():
    # here we are applying the grid_search hyperparameter tuning method to a logistical regression model
    # the hyperparameters we are tuning are as follows: gamma, and kernel
    # the training set is split into the a training and validation part by the GridSearchCV method
    # the hyperparameter tuning part takes roughly 1 hour and 13 minutes to run
    parameters = { #'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], 
              'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
              #'kernel': ['rbf', 'sigmoid', 'poly']}
              'kernel': ['poly']}

    grid_search = GridSearchCV(SVC(), parameters, verbose = 1, cv=3, n_jobs = -1)
    grid_results = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))

findOptimalSVMParameters()

"""
rbf: Best: 0.871431 using {'C': 10000000, 'gamma': 0.0001, 'kernel': 'rbf'}
sigmoid: Best: 0.875939 using {'C': 10000000, 'gamma': 0.0001, 'kernel': 'sigmoid'}
poly
"""