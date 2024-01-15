import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from scipy.stats import boxcox, skew, yeojohnson
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

#### Load data
df = pd.read_csv('MachineLearning\FIFA-2019.csv', index_col=[0])

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
scaled_df = pd.DataFrame(scaled, columns = df.columns)

# Writing the new preprocessed dataset
#scaled_df.to_csv('MachineLearning\editedFIFA.csv', index = False)

# Distribution of skewed features
skewed_features = ['Value', 'Wage', 'Weight']
wage_features = ['Wage']
skewed_df = df[skewed_features]
log_df = np.log(skewed_df.astype(float))

# Yeo-Johnson transforming skewed features
yj_transformer = PowerTransformer(method='yeo-johnson')
yj_transformer.fit(skewed_df)
yj_transformed = yj_transformer.fit_transform(skewed_df)
yj_df = pd.DataFrame(yj_transformed, columns = skewed_df.columns)

# Quantile transforming skewed features
q_transformer = QuantileTransformer()
q_transformer.fit(skewed_df)
q_transformed = q_transformer.fit_transform(skewed_df)
q_df = pd.DataFrame(q_transformed, columns = skewed_df.columns)


for feature in skewed_features:
    #Showing skewed feature
    plt.hist(df[feature])
    plt.xlabel(feature)
    plt.ylabel('frequency')
    plt.show()
    print(feature + " skew: " + str(df[feature].skew()))
    #Showing aforementioned feature log transformed
    plt.hist(log_df[feature])
    plt.xlabel('log_'+feature)
    plt.ylabel('frequency')
    plt.show()
    print(feature + " skew when log transformed: " + str(skew(log_df[feature])))
    #Showing aforementioned feature box-cox transformed
    boxcox_data, boxcox_lambda = boxcox(df[feature].astype(float))
    plt.hist(boxcox_data)
    plt.xlabel('box-cox_' + feature + ' lambda=' + str(boxcox_lambda))
    plt.ylabel('frequency')
    plt.show()
    print(feature + " skew when box-cox transformed: " + str(skew(boxcox_data)))

# used to determine the bins to classifier
mean = scaled_df['Release Clause'].mean()
std = scaled_df['Release Clause'].std()


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
for ind in scaled_df.index:
    releaseClauseRow = scaled_df["Release Clause"][ind]

    if releaseClauseRow < classifier1:
        scaled_df["Release Clause"][ind] = "Really_Cheap"

    elif releaseClauseRow < classifier2:
        scaled_df["Release Clause"][ind] = "Fairly_Cheap"

    elif releaseClauseRow < classifier3:
        scaled_df["Release Clause"][ind] = "Below_Average"

    elif releaseClauseRow < classifier4:
        scaled_df["Release Clause"][ind] = "Above_Average"

    elif releaseClauseRow < classifier5:
        scaled_df["Release Clause"][ind] = "Fairly_Expensive"

    else:
        scaled_df["Release Clause"][ind] = "Really_Expensive"

print(scaled_df["Release Clause"].value_counts())

# split into train test sets
X = scaled_df[['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work_Rate_Low/ High', 'Work_Rate_Low/ Medium', 'Work_Rate_High/ Low', 'Work_Rate_Medium/ Low',
        'Work_Rate_Medium/ Medium', 'Work_Rate_High/ High', 'Work_Rate_High/ Medium', 'Work_Rate_Medium/ High', 'Work_Rate_Low/ Low', 'Body_Type_Lean', 'Body_Type_Normal', 'Body_Type_Stocky', 'Height', 'Weight', 'Crossing', 'Finishing',
        'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',	'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
        'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y = scaled_df['Release Clause']

# creating our train, and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)


# After hyper parameter tuning(values from assignment 2), optimal parameters were shown as below
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=11, weights='distance')
# Fitting the model with data
knn.fit(X_train, y_train)

y_test_knn = knn.predict(X_test)

# Evaluating the accuracy of our KNN model
print('Test set accuracy: ', metrics.accuracy_score(y_test, y_test_knn))

# fit the model with data
clfNaiveBayes = GaussianNB()
clfNaiveBayes.fit(X_train, y_train)

print("Score: ", clfNaiveBayes.score(X_test, y_test))

