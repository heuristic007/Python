'''
Created on Aug 4, 2017
http://xperimentallearning.blogspot/2017/04/scikit-learn-sklearn-library-machine.html
DataCamp - Unsupervised Learning

@author: Yihpyng Kuan
'''
print(__doc__)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

samples = [[ 15.26  ,  14.84  ,   0.871 , 3.312 ,   2.221 ,   5.22  ],
       [ 14.88  ,  14.57  ,   0.8811, 3.333 ,   1.018 ,   4.956 ],
       [ 14.29  ,  14.09  ,   0.905 ,   3.337 ,   2.699 ,   4.825 ], 
       [ 13.2   ,  13.66  ,   0.8883,   3.232 ,   8.315 ,   5.056 ],
       [ 11.84  ,  13.21  ,   0.8521,   2.836 ,   3.598 ,   5.044 ],
       [ 12.3   ,  13.34  ,   0.8684,   2.974 ,   5.637 ,   5.063 ]]

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
