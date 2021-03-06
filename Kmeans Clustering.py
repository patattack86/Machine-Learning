I am applying unsupervised machine learning methods onto my thesis dataset, I have three targets and three parameters 
which we will use to estimate the targets.  


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#reading in some thesis data

data = pd.read_csv("F:/Thesis/Pandas_csv/All_data.csv", sep = ",")

#setting the data up for the algorithm, we're trying to use the three parameters TSS, TP, and Chl-conc 
to estimate the zone we sampled in

mldata = data[['TSS', 'TP', 'Chl-conc', 'Zone']]
mldata = mldata.dropna()

X = mldata[['TSS', 'TP', 'Chl-conc]]
y = mldata['Zone']



#I only have three targets, which are the three different zones that we sampled in. I am trying to predict which zone we sampled in based 
on the values of the parameters at a sampling point, there are 26 sampling points total, in three different zones. 

kmeans = KMeans(n_clusters=3, random_state = 5)
kmeans.fit(X)

#plotting our model output


#this first scatter plot is colored according to the actual species labels, we will use this to compare with the predict species labels

color_theme = np.array(['green', 'blue', 'red'])
plt.subplot(1,2,1)
plt.scatter(x=mldata.TP, y=mldata.TSS, c=color_theme[mldata.Zone], s=50)
plt.title('Ground Truth Classification')

#the difference between these plots is that the colors will be applied according to what we estimated through our
kmeans fit model

y = mldata['Chl-Conc']
x = mldata['TP']
zone = mldata2['Zone']

fig=plt.figure(figsize = (10,10))
ax = plt.gca()
plt.scatter(x , y, c=color_theme[clustering.labels_],  s=50)
plt.ylabel('Chl-a', fontsize = 20)
plt.xlabel('Total Phosphorus', fontsize = 20)
plt.title('K-Means Cluster Analysis', fontsize = 25)
ax.tick_params(labelsize = 15)

for i, txt in enumerate(zone):
    ax.annotate(txt, (x[i],y[i]))
    
plt.show()


#Next step is to evaluate the model to assess performance accuracy.  
print(classification_report(y, X)

