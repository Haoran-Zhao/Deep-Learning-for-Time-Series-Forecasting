import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler

seed = 7
np.random.seed(seed)


def split_dataframe(dataframe):
    train_set = dataframe[dataframe['class'] > 2].values
    test_set = dataframe[(dataframe['class'] < 3) & (dataframe['class'] > 0)].values
    impute_set = dataframe[dataframe['class'] == 0].values
    return train_set, test_set, impute_set


def SVR_training_imputation(dataframe, target, features, output, gamma, C):
    data_train, data_test, data_impute = split_dataframe(dataframe)
    X_train = data_train[:, features]
    scaler = StandardScaler()
    scaler.fit(X_train.astype(float))
    X_train_scale = scaler.transform(X_train.astype(float))
    Y_train = data_train[:, output]
    X_test = data_test[:, features]
    X_test_scale = scaler.transform(X_test.astype(float))
    Y_test = data_test[:, output]
    clf = svm.SVR(kernel='rbf', gamma=gamma, C=C)
    clf.fit(X_train_scale, Y_train)
    y_pred = clf.predict(X_test_scale)
    print("Accuracy:", metrics.r2_score(Y_test, y_pred))

    if target == 'BORE_OIL_VOL':
        for i in range(0, len(dataframe)):
            if dataframe.loc[i, 'BORE_OIL_VOL'] == 0:
                X_impute = dataframe.values[i, features]
                X_impute_scale = scaler.transform(X_impute.reshape(1, -1).astype(float))
                dataframe.at[i, target] = clf.predict(X_impute_scale)
    else:
        for i in range(0, len(dataframe)):
            if dataframe.loc[i, 'class'] == 0:
                X_impute = dataframe.values[i, features]
                X_impute_scale = scaler.transform(X_impute.reshape(1, -1).astype(float))
                dataframe.at[i, target] = clf.predict(X_impute_scale)







# load data
dataframe_dp_tubing = pd.read_excel("datasets.xlsx", sep='delimiter', header=0, sheet_name='dp_tubing')
dataframe_dh_temp = pd.read_excel("datasets.xlsx", sep='delimiter', header=0, sheet_name='dp_tubing')
dataframe_dh_press = pd.read_excel("datasets.xlsx", sep='delimiter', header=0, sheet_name='dp_tubing')
dataframe_annulus_press = pd.read_excel("datasets.xlsx", sep='delimiter', header=0, sheet_name='dp_tubing')
dataframe_whole_data = pd.read_excel("datasets.xlsx", sep='delimiter', header=0, sheet_name='dp_tubing')
print(dataframe_dp_tubing.keys())

for i in range(0, len(dataframe_annulus_press)):
    if np.isnan(dataframe_annulus_press.loc[i, 'AVG_ANNULUS_PRESS']):
        dataframe_annulus_press.at[i, 'class'] = 0
# train_dp_tubing, test_dp_tubing, impute_dp_tubing = split_dataframe(dataframe_dp_tubing)
# train_dh_temp, test_dh_temp, impute_dh_temp = split_dataframe(dataframe_dp_tubing)
# train_dh_press, test_dh_press, impute_dh_press = split_dataframe(dataframe_dp_tubing)
# train_annulus_press, test_annulus_press, impute_annulus_press = split_dataframe(dataframe_dp_tubing)
# train_whole_data, test_whole_data, impute_whole_data = split_dataframe(dataframe_dp_tubing)

SVR_training_imputation(dataframe_dp_tubing, 'AVG_DP_TUBING', [2, 7, 8, 9, 10], 5, 4, 4)

dataframe_dh_temp.loc[:, 'AVG_DP_TUBING'] = dataframe_dp_tubing.loc[:, 'AVG_DP_TUBING']
SVR_training_imputation(dataframe_dh_temp, 'AVG_DOWNHOLE_TEMPERATURE', [2, 5, 7, 8, 9, 10], 4, 5, 3)

dataframe_dh_press.loc[:, 'AVG_DP_TUBING'] = dataframe_dp_tubing.loc[:, 'AVG_DP_TUBING']
SVR_training_imputation(dataframe_dh_press, 'AVG_DOWNHOLE_PRESSURE', [2, 5, 7, 8, 9, 10], 3, 5, 5)


dataframe_annulus_press.loc[:, 'AVG_DP_TUBING'] = dataframe_dp_tubing.loc[:, 'AVG_DP_TUBING']
dataframe_annulus_press.loc[:, 'AVG_DOWNHOLE_PRESSURE'] = dataframe_dh_press.loc[:, 'AVG_DOWNHOLE_PRESSURE']
dataframe_annulus_press.loc[:, 'AVG_DOWNHOLE_TEMPERATURE'] = dataframe_dh_temp.loc[:, 'AVG_DOWNHOLE_TEMPERATURE']
SVR_training_imputation(dataframe_annulus_press, 'AVG_ANNULUS_PRESS', [2, 3, 4, 5, 7, 8, 9, 10], 6, 4, 16)

dataframe_whole_data.loc[:, 'AVG_DP_TUBING'] = dataframe_dp_tubing.loc[:, 'AVG_DP_TUBING']
dataframe_whole_data.loc[:, 'AVG_DOWNHOLE_PRESSURE'] = dataframe_dh_press.loc[:, 'AVG_DOWNHOLE_PRESSURE']
dataframe_whole_data.loc[:, 'AVG_DOWNHOLE_TEMPERATURE'] = dataframe_dh_temp.loc[:, 'AVG_DOWNHOLE_TEMPERATURE']
dataframe_whole_data.loc[:, 'AVG_ANNULUS_PRESS'] = dataframe_annulus_press.loc[:, 'AVG_ANNULUS_PRESS']
SVR_training_imputation(dataframe_whole_data, 'BORE_OIL_VOL', [2, 3, 4, 5, 6, 7, 8, 9, 10], 11, 0.25, 16)

print('done!')
