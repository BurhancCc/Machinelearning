import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


#### Load data
df = pd.read_csv('Machinelearning\FIFA-2019.csv', header=0)

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

limited_scaled_df = scaled_df[[ "Age", "Height", "Weight", "Strength", "Aggression", "Vision"]]

# Testing different amount of clusters to look at the average distance to the centroids
centroid_list = list()
for i in range(1, 16):
    kmeans = KMeans( init="random", n_clusters=i, n_init=10, max_iter=300, random_state=42 )
    kmeans.fit(limited_scaled_df)
    centroid_list.append(kmeans.inertia_)
#    print("Average distance to centroid: %f" %(kmeans.inertia_))

# print(centroid_list)

# Testing different amount of clusters to look at the log-likelihood
cluster_amounts = np.arange(1, 16)
models = [GaussianMixture(n_components=c, n_init=10, max_iter=300, random_state=42).fit(limited_scaled_df) for c in cluster_amounts]
log_likelihoods = [m.score(limited_scaled_df) for m in models]

"""
# a graph showing k the elbow graph to a pick the amount of clusters for K-means
plt.style.use("fivethirtyeight")
plt.plot(range(1, 16), centroid_list)
plt.xticks(range(1, 16))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# Graph showing the elbow graph to pick the amount of clusters for EM
plt.plot(cluster_amounts, log_likelihoods, label="Log-likelihood")
plt.legend()
plt.xticks(cluster_amounts)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.show()
"""

# this method gives us the elbow point for K-means
kl = KneeLocator( range(1, 16), centroid_list, curve="convex", direction="decreasing" )
# here we print the value for the elbow point
print("The amount of clusters for the K-means elbow point = " + str(kl.elbow) + " clusters")
# This method gives us the elbow points EM
em_kl = KneeLocator(cluster_amounts, log_likelihoods, curve="concave", direction="increasing")
# Here we print the values for the EM elbow points
print("The amount of clusters for the EM elbow point = " + str(em_kl.elbow) + " clusters")

# For K-means, the KneeLocator tells us that 4 clusters is optimal here and the eyetest says 3 or 4 clusters.
# For EM, the KneeLocator tells us that 6 clusters is optimal here and the eyetest says 4 to 6 clusters.

"""
# Writing the new preprocessed dataset for checking purposes
scaled_df.to_csv('D:\School\MachineLearning\editedFIFA.csv', index = False)
"""
 
# Initiate k-means, EM, and dbscan algorithms
kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300, random_state=42)
em_cluster_amount = em_kl.elbow
em_model = GaussianMixture(n_components=em_cluster_amount, n_init=10, max_iter=300, random_state=42)

# Fit the algorithms to the features
kmeans.fit(limited_scaled_df)
em_model.fit(limited_scaled_df)

#This step is required to get EM clustering labels
em_labels = em_model.predict(limited_scaled_df)
#Predicted probabilities per entry in dataframe
predicted_probabilities = em_model.predict_proba(limited_scaled_df)
#Predicted probabilities converted into a list for easier editing
predicted_probabilities_list = [[probability for probability in entry] for entry in predicted_probabilities]
#Row locations of entries who were clustered with a high predicted probability
high_probability_indexes = [predicted_probabilities_list.index(x) for x in predicted_probabilities_list if max(x) > 0.7]
high_probability_percentage = len(high_probability_indexes) / len(limited_scaled_df) * 100
print(str(high_probability_percentage) + "% " + "of entries were predicted with a probability of at least 70%")
#Manually made clusters containing values of entries with a high predicted probability
high_probability_clusters = [[[x for x in limited_scaled_df.iloc[i].values] for i in high_probability_indexes if em_labels[i] == cluster] for cluster in range(em_cluster_amount)]
#Means of features in list per cluster in high_probability_clusters
high_probability_means = [[sum([entry[i] for entry in cluster]) / len([entry[i] for entry in cluster]) for i in range(em_cluster_amount)] for cluster in high_probability_clusters]

print(kmeans.labels_)
print(kmeans.cluster_centers_)

#K-means heatmap
fig, ax = plt.subplots()
column_names=["Age", "Height", "Weight", "Strength", "Aggression", "Vision"]
sn.heatmap(pd.DataFrame(kmeans.cluster_centers_), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.set_xticks(np.arange(len(column_names)), labels=column_names)
plt.xlabel('column names')
plt.show()

#EM heatmap
fig, ax = plt.subplots()
sn.heatmap(pd.DataFrame(high_probability_means), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.set_xticks(np.arange(len(column_names)), labels=column_names)
plt.xlabel('column names')
plt.show()

# Plot the data and cluster 
fig, ( ax1, ax2) = plt.subplots( 1, 2, figsize=( 16, 16), sharex = True, sharey = True )
fig.suptitle( f"Clustering Algorithm Comparison: Crescents", fontsize = 16)
fte_colors = { 0: "#A0A000", 1: "#00A0A0", 2: "#A000A0", 3: "#00FF00", -1: "#0000FF", 4: "#ff0000", 5: "#ffa500"}

# The k-means plot
print(type(scaled))
print(scaled[:,1])
print(scaled[:,4])
km_colors = [ fte_colors[label] for label in kmeans.labels_ ]
ax1.scatter( scaled[:, 1], scaled[:, 3], c=km_colors, vmax = 0.8 )
ax1.set_title( f"k-means", fontdict = { "fontsize": 12 } )

# The EM plot
db_colors = [ fte_colors[label] for label in em_labels ]
ax2.scatter( scaled[:, 1], scaled[:, 3], c=db_colors, vmax = 0.8 )
ax2.set_title( f"EM", fontdict = { "fontsize": 12 } )

plt.show()