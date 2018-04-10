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

#import Beautiful Soup
import bs4 as bs
#pickle - Python object serialization
import pickle
# userequests to grab the source code from Wikipedia's page.
import requests

def save_sp500_tickers0():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

save_sp500_tickers()

