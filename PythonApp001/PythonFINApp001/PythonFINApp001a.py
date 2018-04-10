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
Remote Data Access
Functions from pandas_datareader.data and pandas_datareader.wb extract 
data from various Internet sources into a pandas DataFrame. Currently 
the following sources are supported:

Yahoo! Finance
Google Finance
Enigma
Quandl
St.Louis FED (FRED)
Kenneth French's data library
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
#plt.show()

print("use quandl to get MSFT stock info")
dfq = quandl.get('WIKI/MSFT',start_date=start, end_date=end)
dfq.to_csv('MSFT.csv')
dfqc = pd.read_csv('MSFT.csv', parse_dates=True, index_col=0)
dfqc.head()
print(dfqc.head())

from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

dfq_ohlc = dfq['Adj. Close'].resample('10D').ohlc()
dfq_volume = dfq['Volume'].resample('10D').sum()
print(dfq_ohlc.head())

#Since we're just going to graph the columns in Matplotlib, 
#we actually don't want the date to be an index anymore, so we can do:
dfq_ohlc = dfq_ohlc.reset_index()

#Now dates is just a regular column. Next, we want to convert it:
dfq_ohlc['Date'] = dfq_ohlc['Date'].map(mdates.date2num)

#setup the figure:
fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date()

#graph the candlestick graph:
candlestick_ohlc(ax1, dfq_ohlc.values, width=2, colorup='g')
ax2.fill_between(dfq_volume.index.map(mdates.date2num),dfq_volume.values,0)

plt.show()
