'''
Created on Jul 7, 2017
DataSet Source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

@author: Yihpyng Kuan
'''
from sklearn.datasets import load_iris
iris = load_iris()
iris
iris.keys()
#print(iris['DESCR'][:1000] + "\n...")
print(iris['DESCR'] + "\n...")

iris['target_names']
print (iris['target_names'])
iris['feature_names']
print (iris['feature_names'])

print(type(iris['data']))
print(iris['data'].shape)
print(iris['data'][:5])
print ("")
print(iris['data'][:12])
print("")
print(iris['target_names'])
print("")
print(iris['feature_names'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
random_state=0)
#matplotlib inline
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")

for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=25)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i, j].set_visible(False)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
iris['target_names'][prediction]
y_pred = knn.predict(X_test)
np.mean(y_pred == y_test)
print(np.mean(y_pred == y_test))
#accuracy=knn.score(X_test, y_test)
knn.score(X_test, y_test)
print(knn.score(X_test, y_test))

