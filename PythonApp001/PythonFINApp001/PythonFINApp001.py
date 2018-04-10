'''
Created on Aug 9, 2017
http://xperimentallearning.blogspot/2017/04/scikit-learn-sklearn-library-machine.html
DataCamp - Unsupervised Learning

@author: Yihpyng Kuan
'''
'''
https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/
Required Modules to start:

Numpy
Matplotlib
Pandas
Pandas-datareader
BeautifulSoup4
scikit-learn / sklearn
'''
'''
http://pandas-datareader.readthedocs.io/en/latest/remote_data.html
https://www.quandl.com/tools/python
Remote Data Access
Functions from pandas_datareader.data and pandas_datareader.wb extract 
data from various Internet sources into a pandas DataFrame. Currently 
the following sources are supported:

Yahoo! Finance
Google Finance
Enigma
Quandl
St.Louis FED (FRED)
Kenneth French’s data library
World Bank
OECD
Eurostat
Thrift Savings Plan
Nasdaq Trader symbol definitions
'''
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

import quandl

style.use('ggplot')

start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)
end = dt.datetime.now()
#df = web.DataReader('TSLA', "yahoo", start, end) -yahoo is broken
df = web.DataReader('NVDA', "google", start, end)

df.to_csv('NVDA.csv')
dfc = pd.read_csv('nvda.csv', parse_dates=True, index_col=0)
df.plot()
plt.show()

df['Close'].plot()
plt.show()

df[['High','Low', 'Close']].plot()
plt.show()

dfq = quandl.get('WIKI/GOOG') # no start/end dates 
dfq = quandl.get('WIKI/MSFT',start_date=start, end_date=end)

dfq['100ma'] = dfq['Adj. Close'].rolling(window=100,min_periods=0).mean()
print(dfq.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)

ax1.plot(dfq.index, dfq['Adj. Close'])
ax1.plot(dfq.index, dfq['100ma'])
ax2.bar(dfq.index, dfq['Volume'])
plt.show()


'''
https://pythonprogramming.net/stock-data-manipulation-python-programming-for-finance/?completed=/handling-stock-data-graphing-python-programming-for-finance/
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()

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
