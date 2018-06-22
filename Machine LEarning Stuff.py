Importing Libraries

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

#reading and formating data
data = pd.read_csv("F:/Thesis/Pandas_csv/All_data.csv", sep = ",")

mldata = data[['TSS', 'TP', 'Zone']]

mldata = mldata.dropna()

X = mldata[['TSS', 'TP']]

y = mldata['Zone']


#looping to see what value of k produces the most accurate classification
k_range = range(1,31)
k_score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_score.append(scores.mean())

    
plt.plot(k_range, k_score)
plt.show()

#comparing knn with logistic regression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy').mean())
