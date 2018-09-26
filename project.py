import numpy as np
import pandas as pd
from skimage import io
from sklearn import metrics, preprocessing
from matplotlib import pyplot as pl
import additional_stuff

#%%

# Implement the euclidean_distance function here:





# Check if it is working! Here are two vectors:
v1 = np.array([1.1, 2.5, 4.4, 0.1, 2.3, 3.4])
v2 = np.array([2.0, 2.2, 1.0, 1.0, 2.5, 3.4])

# Call the euclidean_distance function below to compute the distance between them:
dist = euclidean_distance(v1, v2)

print('Distance between v1 and v2: ', dist)  # (For these vectors, it should be something like 3.6482...)


#%%

# The Euclidean distance function can be used for k-Means clustering:
def k_means(X, k=3, n_iterations=10):
    # X is a N-by-M numpy array of N data points, each with M dimensions/features
    # k is the number of clusters to compute (3 is the default value)
    # n_iterations is the maximum number of iterations we want (100 is the default value)
    
    # First, we need to initialise the clusters.
    # To simplify, we will use the MacQueen method, selecting 'k' random points of 'X' as initial centroids:
    centroids = additional_stuff.mac_queen_initialisation(X, k)
    
    # Keep a history of the centroids' movements:    
    centroids_history = np.zeros((n_iterations+1, k, X.shape[1]))
    centroids_history[0, :, :] = centroids
    
    # This will store the cluster membership of each data point in 'X':
    membership = np.zeros((X.shape[0]))
    
    # The k_means algorithm is iterative. Start a loop here for 'n_iterations':
    
    
    
    
        # In each loop, for each data point:
        
        
        
        
            # Compute the Euclidean distance between the data point and each centroid:
            
            
            
            
        
            # Then, assign each data point to the cluster with the nearest centroid:
            membership[index] = 
        
        # Now, recompute the centroids of each cluster, computing the mean of the cluster data points:
        for cc in range(k):
            centroids[cc] = 
        
        
        
        centroids_history[ii+1, :, :] = centroids
        
    # Finally, return the clustering result:
    return membership, centroids, centroids_history


#%%
    
# Now, to test it, we shall use a simulated dataset:
X, _ = additional_stuff.simulated_dataset()

# We can use matplotlib to visualise the simulated data points:





# Now, we can apply our implementation of 'k-means' to this dataset:
membership, centroids, centroids_history = k_means(X, k=3)


#%%

# Using matplotlib, we can plot the result:
# First, plot the data points:




# Then, plot the final cluster centroids:







# We can also see the movement of each centroid during the iterations:
# Plot the data points again (use alpha to make them transparent):




# And plot each centroid's movement:




# Then, plot the final cluster centroids:





#%%

# We can also apply our k-means algorithm to a real dataset.
# To do that, we can use pandas to import the dataset:





# Pandas allows us to import datasets with headers, data point names, labels, numerical and text features.
# Using Spyder, you can analyse the dataset using the Variable Explorer.
# To use k-means, let's select just two columns of the dataset:





# And perform k-Means clustering with k = 3 (no. of classes)




# Getting and encoding the labels:
le = preprocessing.LabelEncoder()
y = le.fit_transform(data[['class']])
# This will assess the performance of the clustering (1 - best, 0 - worst)
print(metrics.adjusted_mutual_info_score(y, membership))

# And plot the results:







#%%

# Furthermore, k-means is also used in Image Analysis, for segmentation of images.
# Let's try to apply the same algorithm to an image.

# First, we use scikit-image to import the image:




# Visualise the image using matplotlib:





# Each pixel should be considered a single sample (in this case, with three
# features - RGB). We need to restructure the image accordingly:





# Now, we finally apply k-means to it:





# Reshaping and visualising the result:





# Save the result as an image:




