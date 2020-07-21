import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

# data load
dataframe = pd.read_excel("cleaned_imputed_Oil_production.xlsx", sep='delimiter', header=0)
print(dataframe.keys())

dataset = dataframe.values
X = dataset[:, 2:11]
Y = dataset[:, 11]

scaler = MinMaxScaler()
scaler.fit(X.astype(float))
X_scale = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, random_state=seed)

clf = svm.SVR(kernel='rbf', gamma=1, C=16)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.r2_score(y_test, y_pred))

