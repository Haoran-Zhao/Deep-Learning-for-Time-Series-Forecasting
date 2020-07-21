import numpy as np
import pandas as pd
from keras import backend
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import matplotlib

seed = 7

#data load
dataframe = pd.read_excel("Volve_production_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())
dataframe = dataframe.drop(columns=['AVG_CHOKE_UOM', 'BORE_WI_VOL'])

# self defined functions

# find index of outliers
def find_index(data, outliers, key):
    index = []
    for val in outliers:
        a = data.index[(data[key] == val)].tolist()
        index.append(a[0])
    return index


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


# calculate missing ratio
def missing_ratio(data):
    data_missing = data.isna()
    num_missing = data_missing.sum()
    percentage = num_missing/len(data)
    print(percentage)


# calculate zeros ratio
def zeros_ratio(data):
    data_zeros = data == 0
    num_zeros = data_zeros.sum()
    percentage = num_zeros/len(data)
    print(percentage)


# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())


# production data and injection data
row_production = dataframe[dataframe['FLOW_KIND'] == 'production']
row_injection = dataframe[dataframe['FLOW_KIND'] == 'injection']

# seperate by production threshold
production_over_thre = row_production[row_production['BORE_OIL_VOL'] > 10]
production_less_thre = row_production[row_production['BORE_OIL_VOL'] <= 10]

# drop nan less than 5% of production > 10
production_over_thre = production_over_thre.dropna(subset=['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P', 'AVG_WHP_P'])
print('missing  ratio:')
missing_ratio(production_over_thre)
print('zeros ratio:')
zeros_ratio(production_over_thre)

production_over_thre = production_over_thre.reset_index(drop=True)

# replace zeros in 'ON_STREAM_HRS' and 'AVG_DP_TUBING'
production_over_thre['ON_STREAM_HRS'].replace(to_replace=0, value=np.nan, inplace=True)
production_over_thre['ON_STREAM_HRS'].interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

production_over_thre['AVG_DP_TUBING'].replace(to_replace=0, value=np.nan, inplace=True)
production_over_thre['AVG_DP_TUBING'].interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
# print('Zeros after imputed:')
# zeros_ratio(production_over_thre)
# print('Missing after imputed:')
# missing_ratio(production_over_thre)
# combine dataframe
# df_production = production_over_thre.append(production_less_thre).sort_index()

# data impute for 'AVG_DOWNHOLE_PRESSURE' and 'AVG_DOWNHOLE_TEMPERATURE'


# method1: overall mean
# calculate overall mean for 'AVG_DOWNHOLE_PRESSURE'
df_press = production_over_thre.copy()
index_pressure_0 = df_press.index[df_press['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
df_press = df_press.drop(index_pressure_0)
detect_drop_outlier(df_press, ['AVG_DOWNHOLE_PRESSURE'])
mean_press = df_press['AVG_DOWNHOLE_PRESSURE'].mean()

# calculate overall mean for 'AVG_DOWNHOLE_TEMPERATURE'
df_temp = production_over_thre.copy()
index_temp_0 = df_temp.index[df_temp['AVG_DOWNHOLE_TEMPERATURE'] == 0].tolist()
df_temp = df_temp.drop(index_temp_0)
detect_drop_outlier(df_temp, ['AVG_DOWNHOLE_TEMPERATURE'])
mean_temp = df_temp['AVG_DOWNHOLE_TEMPERATURE'].mean()

# impute overall mean for 'AVG_DOWNHOLE_PRESSURE' and 'AVG_DOWNHOLE_TEMPERATURE'
df_overall_mean = production_over_thre.copy()
df_overall_mean = df_overall_mean.replace({'AVG_DOWNHOLE_PRESSURE': {0: mean_press}})
df_overall_mean = df_overall_mean.replace({'AVG_DOWNHOLE_TEMPERATURE': {0: mean_temp}})
df_overall_mean.to_excel("overall_mean_data.xlsx", index=False)


# method2: MLP predict 'AVG_DOWNHOLE_PRESSURE' and 'AVG_DOWNHOLE_TEMPERATURE'
# load 'AVG_DOWNHOLE_PRESSURE' model
press_json_file = open('model_MLP_PRESS.json', 'r')
loaded_press_model = press_json_file.read()
press_json_file.close()
press_model = model_from_json(loaded_press_model)
press_model.load_weights('model_MLP_PRESS.h5')
print('Loaded pressure model form disk')
press_model.compile(loss='mse', optimizer='adam')

# load 'AVG_DOWNHOLE_PRESSURE' model
temp_json_file = open('model_MLP_TEMP.json', 'r')
loaded_temp_model = temp_json_file.read()
temp_json_file.close()
temp_model = model_from_json(loaded_temp_model)
temp_model.load_weights('model_MLP_TEMP.h5')
print('Loaded temperature model form disk')
temp_model.compile(loss='mse', optimizer='adam')

df_MLP = production_over_thre.copy()
df_press_1 = production_over_thre.copy()
df_temp_1 = production_over_thre.copy()
X = df_MLP.loc[:, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
index_press_zeros = df_MLP.index[df_MLP['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
index_temp_zeros = df_MLP.index[df_MLP['AVG_DOWNHOLE_TEMPERATURE'] == 0].tolist()

scaler = MinMaxScaler()
scaler.fit(X.astype(float))

for value in index_press_zeros:
    input_value = df_press_1.loc[value, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
    input_value = input_value.values.reshape(1, 5)
    x = scaler.transform(input_value)
    df_press_1.loc[value, 'AVG_DOWNHOLE_PRESSURE'] = press_model.predict(x)[0]

for value in index_temp_zeros:
    input_value = df_temp_1.loc[value, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
    input_value = input_value.values.reshape(1, 5)
    x = scaler.transform(input_value)
    df_temp_1.loc[value, 'AVG_DOWNHOLE_TEMPERATURE'] = temp_model.predict(x)[0]

df_MLP.loc[:, 'AVG_DOWNHOLE_PRESSURE'] = df_press_1.loc[:, 'AVG_DOWNHOLE_PRESSURE']
df_MLP.loc[:, 'AVG_DOWNHOLE_TEMPERATURE'] = df_temp_1.loc[:, 'AVG_DOWNHOLE_TEMPERATURE']
print('Zeros_ratio_MLP after:')
zeros_ratio(df_MLP)
print('Missing_Ratio_MLP after:')
missing_ratio(df_MLP)
df_MLP.to_excel("MLP_data.xlsx", index=False)


# method3: SVR
# Load from file
SVR_temp = pickle.load(open('SVR_temperature.sav', 'rb'))
SVR_press = pickle.load(open('SVR_pressure.sav', 'rb'))

df_SVR = production_over_thre.copy()
df_press_2 = production_over_thre.copy()
df_temp_2 = production_over_thre.copy()
print('SVR_missing_ratio:')
missing_ratio(df_SVR)
print('SVR_zeros_ratio:')
zeros_ratio(df_SVR)
X = df_SVR.loc[:, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
index_press_zeros = df_SVR.index[df_SVR['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
index_temp_zeros = df_SVR.index[df_SVR['AVG_DOWNHOLE_TEMPERATURE'] == 0].tolist()

scaler = MinMaxScaler()
scaler.fit(X.astype(float))

for value in index_press_zeros:
    input_value = df_press_2.loc[value, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
    input_value = input_value.values.reshape(1, 5)
    x = scaler.transform(input_value)
    df_press_2.at[value, 'AVG_DOWNHOLE_PRESSURE'] = SVR_press.predict(x)

for value in index_temp_zeros:
    input_value = df_temp_2.loc[value, ['ON_STREAM_HRS', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P']]
    input_value = input_value.values.reshape(1, 5)
    x = scaler.transform(input_value)
    df_temp_2.at[value, 'AVG_DOWNHOLE_TEMPERATURE'] = SVR_temp.predict(x)


df_SVR.loc[:, 'AVG_DOWNHOLE_PRESSURE'] = df_press_2.loc[:, 'AVG_DOWNHOLE_PRESSURE']
df_SVR.loc[:, 'AVG_DOWNHOLE_TEMPERATURE'] = df_temp_2.loc[:, 'AVG_DOWNHOLE_TEMPERATURE']
print('Zeros_ratio of SVR:')
zeros_ratio(df_SVR)
print('Missing ratio_SVR after')
missing_ratio(df_SVR)
df_SVR.to_excel("SVR_data.xlsx", index=False)








