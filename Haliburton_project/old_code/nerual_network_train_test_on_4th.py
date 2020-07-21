import seaborn as sns
import datetime
from pandas import to_datetime
import numpy as np
from keras import backend
from keras.models import model_from_json
from statistics import mean
import math
import xlrd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates

# font = {'size': 16}
# matplotlib.rc('font', **font)

np.seterr(divide='ignore', invalid='ignore')

seed = 7
# data load
dataframe = pd.read_excel("Volve_production_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())

'''
self defined function
'''


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

        # print(outliers)
        # outlier_index = find_index(data_1, outliers, i)
        # print(outlier_index)
        # data_1[i].drop(outlier_index)


# IQR method calculate outliers
def IQR(data_1):
    sorted(data_1)
    q1, q3 = np.percentile(data_1, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * q1)
    if lower_bound < 0:
        lower_bound = 0
    upper_bound = q3 + (1.5 * q3)
    return iqr, lower_bound, upper_bound

# iqr, lower_bound, upper_bound = IQR(well_p1_1['BORE_OIL_VOL'])
# print(iqr, lower_bound, upper_bound)


# calculate linear regression neural network accuracy
def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))

# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())

# production data and injection data
row_production = dataframe[dataframe['FLOW_KIND'] == 'production']
row_injection = dataframe[dataframe['FLOW_KIND'] == 'injection']

'''
calculate mean values of 'AVG_DOWNHOLE_TEMPERATURE' and 'AVG_DOWNHOLE_PRESS' in all production wells without zeros rows 
1. drop nan in 9 input features
2. drop zeros rows of 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'
3. drop outliers in 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P',  'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'
4. calculate mean of 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'
'''
row_production = row_production.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'DP_CHOKE_SIZE', 'AVG_WHT_P', 'AVG_WHP_P', 'AVG_ANNULUS_PRESS'])

index_pressure_all = row_production.index[row_production['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
index_temp_all = row_production.index[row_production['AVG_DOWNHOLE_TEMPERATURE'] == 0].tolist()
index_unique = list(set().union(index_pressure_all, index_temp_all))
row_production = row_production.drop(index_unique)

detect_drop_outlier(row_production, ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P',  'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'])

mean_pressure = row_production['AVG_DOWNHOLE_PRESSURE'].mean()
mean_temp = row_production['AVG_DOWNHOLE_TEMPERATURE'].mean()

'''
1-5 wells (production), well_6 and well_7(injection), drop missing data in colums which less than 5%, Drop "AVG_CHOKE_UOM" column
'''
# fill zeros in column 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE' with mean value
dataframe['AVG_DOWNHOLE_PRESSURE'] = dataframe.AVG_DOWNHOLE_PRESSURE.mask(dataframe.AVG_DOWNHOLE_PRESSURE == 0, mean_pressure)
dataframe['AVG_DOWNHOLE_TEMPERATURE'] = dataframe.AVG_DOWNHOLE_TEMPERATURE.mask(dataframe.AVG_DOWNHOLE_TEMPERATURE == 0, mean_temp)

# 7 wells box plot (production)
# DROP NAN row less than 5% for production wells
well_p_1 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7405) & (dataframe['FLOW_KIND'] == 'production')]
well_p_1 = well_p_1.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE'])
well_p_1 = well_p_1.drop(columns=['AVG_CHOKE_UOM'])


well_p_2 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7078) & (dataframe['FLOW_KIND'] == 'production')]
well_p_2 = well_p_2.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'DP_CHOKE_SIZE', 'AVG_WHT_P', 'AVG_WHP_P', 'AVG_ANNULUS_PRESS'])
well_p_2 = well_p_2.drop(columns=['AVG_CHOKE_UOM'])

well_p_3 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5599) & (dataframe['FLOW_KIND'] == 'production')]
well_p_3 = well_p_3.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS'])
well_p_3 = well_p_3.drop(columns=['AVG_CHOKE_UOM'])

well_p_4 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5351) & (dataframe['FLOW_KIND'] == 'production')]
well_p_4 = well_p_4.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P'])
well_p_4 = well_p_4.drop(columns=['AVG_CHOKE_UOM'])

well_p_5 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7289) & (dataframe['FLOW_KIND'] == 'production')]
well_p_5 = well_p_5.drop(columns=['AVG_CHOKE_UOM'])

list_p = [well_p_1, well_p_2, well_p_3, well_p_4, well_p_5]

# Drop NaN row less than 5% for injection wells
well_p_6 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5693) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_6 = well_p_6.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])
well_p_6 = well_p_6.drop(columns=['AVG_CHOKE_UOM'])

well_p_7 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5769) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_7 = well_p_7.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])
well_p_7 = well_p_7.drop(columns=['AVG_CHOKE_UOM'])

list_i = [well_p_6, well_p_7]

for data in list_p:
    detect_drop_outlier(data, ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'])

dataset_4 = well_p_4.dropna(subset=['AVG_ANNULUS_PRESS']).values
X = dataset_4[:, [8, 9, 10, 11, 13, 15, 14, 16, 17]]
Y = dataset_4[:, 12]
Y = np.reshape(Y, (-1, 1))
scaler = MinMaxScaler()
print(scaler.fit(X.astype(float)))
xscale = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(xscale, Y, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=[r2_keras])
history = model.fit(X_train, Y_train, epochs=1000, batch_size=50,  verbose=1, validation_split=0.2)

score = model.evaluate(X_test, Y_test)
print(model.metrics_names)
print('mse: %.2f' % (score[0]))
print('r2: %.2f%%' % (score[1]*100))
# print(history.history.keys())

# serialize model to JSON
model_json = model.to_json()
with open("model_train_test_on_4th.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_train_test_on_4th.h5")
print("Saved model to disk")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
