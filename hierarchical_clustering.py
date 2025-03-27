# shuiming chen
# 03/25/2025


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


# load the data from csv file and only keep three features
data = pd.read_csv("./Data/Spotify_Youtube.csv")
df = data[["Liveness", "Energy", "Loudness"]]

# Normalize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Hierarchical Clustering
linkage_matrix = linkage(data_scaled, method='ward')

# draw the map of Hierarchical Clustering
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90, leaf_font_size=8)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()