import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as  plt

seed = 7
np.random.seed(seed)

# load data
dataframe = pd.read_excel("overall_mean_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())


# calculate missing ratio
def missing_ratio(data):
    data_missing = data.isna()
    num_missing = data_missing.sum()
    percentage = num_missing/len(data)
    print(percentage)


df = dataframe.copy()
# drop zeros in annulus
index_0 = df.index[df['AVG_ANNULUS_PRESS'] == 0].tolist()
index_nan = df['AVG_ANNULUS_PRESS'].index[df['AVG_ANNULUS_PRESS'].apply(np.isnan)].tolist()
df = df.drop(index_0)
df = df.drop(index_nan)
missing_ratio(df)
dataset_value = df.values
X = dataset_value[:, [8, 9, 10, 11, 13, 14, 15, 16]]
Y = dataset_value[:, 12]

# scaler = MinMaxScaler()
# print(scaler.fit(X.astype(float)))
# Xscale = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed)

clf = svm.SVR(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.r2_score(y_test, y_pred))
