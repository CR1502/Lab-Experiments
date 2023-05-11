# Hierarchical Agglomerative Clustering

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('lab6_1.csv')

# Perform hierarchical agglomerative clustering
clustering = AgglomerativeClustering(n_clusters=3).fit(data)

# Visualize the results
plt.scatter(data['x'], data['y'], c=clustering.labels_)
plt.show()
