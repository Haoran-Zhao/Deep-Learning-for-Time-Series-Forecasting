import seaborn as sns
import numpy as np
from statistics import mean
import math
import xlrd
import pandas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from keras.models import model_from_json
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


np.seterr(divide='ignore', invalid='ignore')

seed = 7
np.random.seed(seed)
# data load
dataframe = pandas.read_excel("Volve_production_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())

# find index of outlier
def find_index(data, outliers, key):
    index = []
    for val in outliers:
        a = data.index[(data[key] == val)].tolist()
        index.append(a[0])
    return index


def fill_data(data):
    a = data.index[data['ON_STREAM_HRS'] == 0].tolist()
    for index in a:
        if np.isnan(data.loc[index, 'AVG_DOWNHOLE_PRESSURE']):
            data.at[index, 'AVG_DOWNHOLE_PRESSURE'] = 0
        if np.isnan(data.loc[index, 'AVG_DOWNHOLE_TEMPERATURE']):
            data.at[index, 'AVG_DOWNHOLE_TEMPERATURE'] = 0
        if np.isnan(data.loc[index, 'AVG_DP_TUBING']):
            data.at[index, 'AVG_DP_TUBING'] = 0
        if np.isnan(data.loc[index, 'AVG_CHOKE_SIZE_P']):
            data.at[index, 'AVG_CHOKE_SIZE_P'] = 0


# detect outlier by z-score
def detect_drop_outlier(data_1, col):
    for i in col:
        threshold = 3
        mean_1 = np.mean(data_1[i])
        std_1 = np.std(data_1[i])

        for y in data_1[i]:
            z_score = (y - mean_1) / std_1
            if np.abs(z_score) > threshold:
                a = data_1.index[data_1[i] == y].tolist()
                data_1.drop(a, inplace=True)


# calculate linear regression neural network accuracy
def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())


# 7 wells box plot (production)
# DROP NAN row less than 5% for production wells
well_p_1 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7405) & (dataframe['FLOW_KIND'] == 'production')]
well_p_1 = well_p_1.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE'])
well_p_1.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

well_p_2 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7078) & (dataframe['FLOW_KIND'] == 'production')]
well_p_2 = well_p_2.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'DP_CHOKE_SIZE', 'AVG_WHT_P', 'AVG_WHP_P', 'AVG_ANNULUS_PRESS'])
well_p_2.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

well_p_3 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5599) & (dataframe['FLOW_KIND'] == 'production')]
well_p_3 = well_p_3.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS'])
well_p_3.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

well_p_4 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5351) & (dataframe['FLOW_KIND'] == 'production')]
well_p_4 = well_p_4.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P'])
well_p_4.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

well_p_5 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7289) & (dataframe['FLOW_KIND'] == 'production')]
well_p_5.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

list_p = [well_p_1, well_p_2, well_p_3, well_p_4, well_p_5]

# Drop NaN row less than 5% for injection wells
well_p_6 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5693) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_6 = well_p_6.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])
well_p_6.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

well_p_7 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5769) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_7 = well_p_7.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])
well_p_7.drop(columns=['AVG_CHOKE_UOM'], inplace=True)

list_i = [well_p_6, well_p_7]

# detect and drop outliers aftering droping less than 5% rows
for data in list_p:
    detect_drop_outlier(data, ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL'])

print(well_p_5.iloc[:, 8:17].head())
dataset_4 = well_p_4.dropna(subset=['AVG_ANNULUS_PRESS'])
X_test_4 = dataset_4.values[:, [8, 9, 10, 11, 13, 15, 16, 17]]
Y_test_4 = dataset_4.values[:, 12]
Y_test_4 = np.reshape(Y_test_4, (-1, 1))

X = dataset_4.values[:, [8, 9, 10, 11, 13, 15, 16, 17]]
Y = dataset_4.values[:, 12]
Y = np.reshape(Y, (-1, 1))
scaler = MinMaxScaler()
print(scaler.fit(X))
print(scaler.fit(Y))
xscale = scaler.transform(X)
yscale = scaler.transform(Y)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='normal', activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=[r2_keras])
history = model.fit(X_train, y_train, epochs=500, batch_size=50,  verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print(model.metrics_names)
print('Test loss: %.2f%%' % (score[0]*100))
print('Test accuracy: %.2f%%' % (score[1]*100))
print(history.history.keys())
score_1 = model.evaluate(X_test_4, Y_test_4, verbose=1)
print('Test_1 loss: %.2f%%' % (score_1[0]*100))
print('Test_1 accuracy: %.2f%%' % (score_1[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model_all_3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_all_3.h5")
print("Saved model to disk")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()




# # define model
# def baseline_model():
#     # creat model
#     model = Sequential()
#     model.add(Dense(10, input_dim=8, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model
#
#
# # evaluate model with standardized dataset
# np.random.seed(seed)
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#
# # estimator.fit(X, Y)
# # prediction = estimator.predict(X)
# # accuracy_score(Y, prediction)