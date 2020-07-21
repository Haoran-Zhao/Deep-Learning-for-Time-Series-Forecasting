import seaborn as sns
from pandas import DataFrame
from pandas import concat
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
print(dataframe.head())


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


def series_to_supervised(data, n_in =1, n_out =1, dropnan =True):
    data=[]


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
