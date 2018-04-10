'''
Created on Jul 29, 2017
DataSet Source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

pd.set_option('display.max_rows', 1000)

Python Data Visualization Cookbook author: Igor Milovanovic

@author: Yihpyng Kuan
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def is_outlier(points, threshold=3.5):
    """
    Returns a boolean array with True if points are outliers and
    False
    otherwise.
    Data points with a modified z-score greater than this
    # value will be classified as outliers.
    """
# transform into vector
    if len(points.shape) == 1:
        points = points[:,None]
# compute median value
        median = np.median(points, axis=0)
# compute diff sums along the axis
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
# compute MAD
        med_abs_deviation = np.median(diff)
# compute modified Z-score
# http://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Iglewicz
        modified_z_score = 0.6745 * diff / med_abs_deviation
# return a mask for each outlier
        return modified_z_score > threshold
# Random data
x = np.random.random(100)
# histogram buckets
buckets = 50
# Add in a few outliers
x = np.r_[x, -49, 95, 100, -100]

# Keep valid data points
# Note here that
# "~" is logical NOT on boolean numpy arrays
filtered = x[~is_outlier(x)]
# plot histograms
plt.figure()
plt.subplot(211)
plt.hist(x, buckets)
plt.xlabel('Raw')
plt.subplot(212)
plt.hist(filtered, buckets)
plt.xlabel('Cleaned')
plt.show()
