import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans as kmeans
from matplotlib import pyplot as plt

# Loading and normalizing data
data = pd.read_csv('iris.txt')
X = np.array(data[['petal width', 'petal length']])
X = preprocessing.scale(X)
le = preprocessing.LabelEncoder()
y = le.fit_transform(data[['class']])

# Creating the object, estimating the parameters and predicting membership
model = kmeans(n_clusters=3)
model.fit(X)
membership = model.predict(X)
centroids = model.cluster_centers_
print("Adjusted mutual info:", metrics.adjusted_mutual_info_score(y,
                                                                  membership))


# Plotting results
for kk in range(X.shape[0]):
    if membership[kk] == 0:
        plt.plot(X[kk, 0], X[kk, 1], 'x', color='red')
    elif membership[kk] == 1:
        plt.plot(X[kk, 0], X[kk, 1], 'x', color='green')
    elif membership[kk] == 2:
        plt.plot(X[kk, 0], X[kk, 1], 'x', color='blue')
plt.plot(centroids[:, 0], centroids[:, 1], 'o', color='black')
plt.title('Clustering Result')
plt.xlabel('Petal width (cm)')
plt.ylabel('Petal length (cm)')
plt.show()
