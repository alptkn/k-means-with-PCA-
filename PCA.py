# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:40:50 2019

@author: 90553
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:03:30 2019

@author: 90553
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer

data = np.asarray(pd.read_csv("train.csv",skiprows = 0))
data_test = np.asarray(pd.read_csv("test.csv",skiprows = 0))
X = data[:,:595]
Y = data[:,595]
sc = StandardScaler()
"""
X_train = sc.fit_transform(X)
X_test  = sc.fit_transform(data_test)
"""

scaler_train = StandardScaler().fit(X)
X_train = scaler.transform(X)

scaler_test = StandardScaler().fit(data_test)
X_test = scaler2.transform(data_test)

     

pca = PCA(n_components = 75)
X_train_clean = pca.fit_transform(X_train)
X_test_clean  = pca.fit_transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train_clean,Y)
y_predict = classifier.predict(X_test_clean)
y_predict.shape = (np.size(y_predict),1)
temp = np.ones((80,1),dtype = float)
for i in range(0,80):
    temp[i] = i + 1;
Y_csv = np.concatenate((temp,y_predict),1)

np.savetxt('Resultn2.csv',Y_csv,delimiter = ",",fmt = '%.0f',header = "ID,Predicted" )


