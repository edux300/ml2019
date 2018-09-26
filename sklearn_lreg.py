# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:55:10 2018

@author: eduardo

sklearn kmeans
sklearn log_reg
keras log_reg
keras log_reg MNIST
keras cnn MNIST


"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt

def plot_fs(predict_func, X, y, resolution = .01, save_name=None):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

TRAINING_SIZE = 0.5
RANDOM_STATE = 22
# LOAD DATA
data = pd.read_csv('iris.txt')
# We only use the first two features
X = np.array(data[['sepal length', 'sepal width']])
le = preprocessing.LabelEncoder()
y = le.fit_transform(data[['class']])


# NORMALIZE DATA
# In most machine learning algorithms it is good practise to scale
# all features so they have the same mean and std (e.g. in k-means
# unscalled features can lead to some of them having more importance 
# than others.
X = scale(X)

# DIVIDE DATA INTO TRAIN AND TEST
# To better access the performance of a method it should be tested 
# on unseen data. For more on this search for "overfitting".
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=TRAINING_SIZE,
                                                    stratify=y,
                                                    random_state=RANDOM_STATE)

# TRAIN A MODEL EstimatorOBJ.fit
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# ESTIMATE THE SCORE ON THE TRAIN AND ON THE TEST SET
print("Train set: ", logreg.score(X_train, y_train))
print("Test set: ", logreg.score(X_test, y_test))

plot_fs(logreg.predict, X, y, save_name="test.png")
