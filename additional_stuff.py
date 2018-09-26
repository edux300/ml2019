import numpy as np
from sklearn import cluster, datasets, mixture

def simulated_dataset():
    return datasets.make_blobs(n_samples=1000, cluster_std=[2.0, 2.5, 1.5], random_state=112)

def mac_queen_initialisation(X, k):
    centroids = np.zeros((k, X.shape[1]))
    for cc in range(k):
        index = np.random.randint(0, high=X.shape[0])
        centroids[cc] = X[index]
    return centroids