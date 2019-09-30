import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

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
                                                    
# One hot encoding
y_ = np.zeros([75,3])
for i, value in enumerate(y_train):
    y_[i, value] = 1
y_train = y_

# One hot encoding
y_ = np.zeros([75,3])
for i, value in enumerate(y_test):
    y_[i, value] = 1
y_test = y_    


model = Sequential()
model.add(Dense(3, input_dim=2, activation='softmax'))
batch_size = 75
nb_epoch = 250

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))

print("Train set: ", model.evaluate(X_train, y_train, verbose=0)[1])
print("Test set: ", model.evaluate(X_test, y_test, verbose=0)[1])

pred_func = lambda x : np.argmax(model.predict(x), axis=1)
plot_fs(pred_func, X, y, save_name="test.png")

