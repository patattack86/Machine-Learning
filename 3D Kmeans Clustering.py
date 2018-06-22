import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3d
import pandas as pd
import numpy as np


#preparing data
data = pd.read("E:/Thesis/Pandas_csv/All_data.csv", sep = ",")

mldata = data[['TSS', 'TP', 'Chl-Conc']]

X = mldata.dropna()

#Initiating Kmeans instance
clustering = KMeans(n_clusters=3)
clustering.fit(X)

centroids = clustering.cluster_centers_
labels = clustering.predict(X)


#plotting clustering results
fig=plt.figure(figsize = (10,10))
ax=Axes3D(fig)
ax.tick_params(labelsize = 12)
ax.scatter(mldata['Chl-Conc'], mldata['TP'], mldata['TSS'], c=color_theme[clustering.labels_])
ax.set_zlabel('TSS', fontsize = 20)
ax.set_xlabel('Chl-a', fontsize = 20)
ax.set_ylabel('TP', fontsize = 20)
plt.show()
