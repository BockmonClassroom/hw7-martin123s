# shuiming chen
# 03/25/2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# load the data from csv file and only keep three features
data = pd.read_csv("./Data/Spotify_Youtube.csv")
df = data[["Liveness", "Energy", "Loudness"]]

# using elbow method of find the optimal k
sse = []
for i in range(1,11): # for efficiency here will try to find k from 1 to 10
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(df)
    sse.append(km.inertia_)

# plot elbow method figure
fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,11), sse, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('SSE')
plt.title('The Elbow Method Result')
plt.show()


# Choose the optimal K (based on the above elbow graph K=3)
k_optimal = 3
kmeans = KMeans(n_clusters = k_optimal, init = 'k-means++',  random_state=42)
df['Cluster'] = kmeans.fit_predict(df)


# 3D Visualization
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection ='3d')
scatter = ax.scatter(df['Liveness'], df['Energy'], df['Loudness'], c=df['Cluster'], 
          cmap='viridis', edgecolors='k')
ax.set_xlabel('Liveness')
ax.set_ylabel('Energy')
ax.set_zlabel('Loudness')
ax.set_title(f'3D Cluster Distribution Visualization (K={k_optimal})')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()