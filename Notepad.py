from operator import index
from re import X
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sqlalchemy import false


#### Load data
df = pd.read_csv('MachineLearning\FIFA-2019.csv', header=0)

# Dropping the columns that are not important
df = df.drop(columns = ["ID", "Name", "Photo", "Nationality", "Flag", "Club", "Club Logo", "Real Face", "Position", "Jersey Number", "Joined", "Loaned From", "Contract Valid Until",
              "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
              "CB", "RCB", "RB", "Body Type", "Preferred Foot", "Work Rate"])

# Cleaning up the relevant columns and dropping the rows with NaN values
df = df.dropna()
df["Value"] = df["Value"].str.replace("€", "")
df["Wage"] = df["Wage"].str.replace("€", "")
df["Release Clause"] = df["Release Clause"].str.replace("€", "")
df["Weight"] = df["Weight"].str.replace("lbs", "")

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
limited_scaled_df = scaled_df[["Age", "Height", "Weight", "Strength", "Aggression", "Vision"]]

em_model = GaussianMixture(n_components=6, n_init=10, max_iter=300, random_state=42)
em_model.fit(limited_scaled_df)

em_labels = em_model.predict(limited_scaled_df)
predicted_probabilities = em_model.predict_proba(limited_scaled_df)
score_samples = em_model.score_samples(limited_scaled_df)
predicted_probabilities_list = [[probability for probability in entry] for entry in predicted_probabilities]

manual_labels = [entry.index(max(entry)) for entry in predicted_probabilities_list]

highScores = [max(x) for x in predicted_probabilities if max(x) > 0.7]

percentage = len(highScores) / len(limited_scaled_df) * 100

print(percentage)

high_probability_indexes = [predicted_probabilities_list.index(x) for x in predicted_probabilities_list if max(x) > 0.7]

high_probability_clusters = [[[x for x in limited_scaled_df.iloc[i].values] for i in high_probability_indexes if em_labels[i] == cluster] for cluster in range(6)]

high_probability_means = [[sum([entry[i] for entry in cluster]) / len([entry[i] for entry in cluster]) for i in range(6)] for cluster in high_probability_clusters]

column_names=["Age", "Height", "Weight", "Strength", "Aggression", "Vision"]
high_probability_columns = [[limited_scaled_df.iloc[i][column] for i in range(len(limited_scaled_df))] for column in column_names]

print(len(high_probability_columns))