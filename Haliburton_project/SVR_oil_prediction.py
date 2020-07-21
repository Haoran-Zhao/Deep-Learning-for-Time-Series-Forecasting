import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler

seed = 7
np.random.seed(seed)


# calculate missing ratio
def missing_ratio(data):
    data_missing = data.isna()
    num_missing = data_missing.sum()
    percentage = num_missing/len(data)
    print(percentage)


def split_dataframe(dataframe):
    train_set = dataframe[dataframe['class'] > 2].values
    test_set = dataframe[(dataframe['class'] < 3) & (dataframe['class'] > 0)].values
    return train_set, test_set


dataframe = pd.read_excel("op_wi.xlsx", sep='delimiter', header=0)
dataframe = dataframe.fillna(0)
# missing_ratio(dataframe)
data_train, data_test = split_dataframe(dataframe)
X_train = data_train[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]]
scaler = StandardScaler()
scaler.fit(X_train.astype(float))
X_train_scale = scaler.transform(X_train.astype(float))
Y_train = data_train[:, 11]
X_test = data_test[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]]
X_test_scale = scaler.transform(X_test.astype(float))
Y_test = data_test[:, 11]
acc_last = 0.0001
clf = svm.SVR(kernel='rbf', gamma=0.17, C=70)
clf.fit(X_train_scale, Y_train)
y_pred = clf.predict(X_test_scale)
print("Accuracy:", metrics.r2_score(Y_test, y_pred))
print('done!')
