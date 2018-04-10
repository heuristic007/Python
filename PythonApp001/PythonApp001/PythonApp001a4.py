'''
Created on Jul 29, 2017
DataSet Source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

pd.set_option('display.max_rows', 1000)

@author: Yihpyng Kuan
'''
import pandas as pd
import matplotlib.pyplot as plt

# Make DataFrame of iris data
import sklearn.datasets
data = sklearn.datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target
df = df.rename(columns={'petal length (cm)': 'petal_length'})
for i in [0, 1, 2]:
    df.loc[df['species']==i, 'species'] = data.target_names[i]

df = df[['species', 'petal_length']]

# Extract Series that has versicolor petal lengths
versicolor_petal_length = df.loc[df.species=='versicolor',
                                 'petal_length'].values
                                 
plt.hist(versicolor_petal_length)
plt.show()
