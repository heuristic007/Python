'''
Created on Aug 4, 2017
http://xperimentallearning.blogspot/2017/04/scikit-learn-sklearn-library-machine.html
DataCamp - Unsupervised Learning

@author: Yihpyng Kuan
'''
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
new_points = np.array([[0, 0], [4, 4]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [4, 4]])
kmeans.cluster_centers_

# Import KMeans
from sklearn.cluster import KMeans 

points = X
# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
#labels = model.predict(new_points)
labels = model.predict([[0, 0], [4, 4]])
print(labels)

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)

plt.show()
