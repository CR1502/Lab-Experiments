# K-Means Clustering

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')

# Perform k-means clustering
clustering = KMeans(n_clusters=3).fit(data)

# Visualize the results
plt.scatter(data['x'], data['y'], c=clustering.labels_)
plt.show()
