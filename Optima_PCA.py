# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:05:43 2019

@author: 90553
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

data = np.asarray(pd.read_csv("train.csv",skiprows = 0))
data_test = np.asarray(pd.read_csv("test.csv",skiprows = 0))
X = data[:,:595]
Y = data[:,595]

scaler = MinMaxScaler(feature_range = [0,1])
data_rescaled = scaler.fit_transform(X)

pca = PCA().fit(data_rescaled)

plt.Figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Variance')
plt.show()