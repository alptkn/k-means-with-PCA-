import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def loadData(trainPath, testPath):  
    data = np.asarray(pd.read_csv("train.csv",skiprows = 0))
    data_test = np.asarray(pd.read_csv("test.csv",skiprows = 0))
    return data, data_test

def preprocessing(data,data_test):
    scaler_train = StandardScaler().fit(X)
    X_train = scaler_train.transform(X)

    scaler_test = StandardScaler().fit(data_test)
    X_test = scaler_test.transform(data_test) 
    return X_train,X_test

def findComponent(X):
    scaler = MinMaxScaler(feature_range = [0,1])
    data_rescaled = scaler.fit_transform(X)
    pca = PCA().fit(data_rescaled)
    plt.Figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Variance')
    plt.show()

def featureProjection(X_train,X_test):
    pca = PCA(n_components = 70)
    X_train_clean = pca.fit_transform(X_train)
    X_test_clean  = pca.fit_transform(X_test)
    return X_train_clean,X_test_clean

def trainModel(X_train_clean, Y):
    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(X_train_clean,Y)
    
    return classifier

def predict(classifier, X_test_clean):
    y_predict = classifier.predict(X_test_clean)
    y_predict.shape = (np.size(y_predict),1)
    return y_predict

def writeOutput(result):
    np.savetxt('Results.csv', result, delimiter = ",", comments='', fmt = '%.0f', header = "ID,Predicted" )
    #result.to_csv("result.csv")

data, data_test = loadData("train.csv",
                            "test.csv")

X = data[:,:595]
Y = data[:,595]


findComponent(X)

X_train,X_test = preprocessing(X,data_test)

X_train_clean,X_test_clean = featureProjection(X_train,X_test)

classifier = trainModel(X,Y)

y_predict = predict(classifier, X_test)

temp = np.ones((80,1),dtype = float)
for i in range(0,80):
    temp[i] = i + 1;
Y_csv = np.concatenate((temp,y_predict),1)

writeOutput(Y_csv)

