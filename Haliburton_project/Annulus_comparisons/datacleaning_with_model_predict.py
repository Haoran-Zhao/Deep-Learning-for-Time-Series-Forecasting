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
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.dates as mdates

# font = {'size': 16}
# matplotlib.rc('font', **font)

np.seterr(divide='ignore', invalid='ignore')

seed = 7
# data load
dataframe = pd.read_excel("Volve_production_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())
dataframe = dataframe.drop(columns=['AVG_CHOKE_UOM', 'BORE_WI_VOL'])

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

# production data and injection data
row_production = dataframe[dataframe['FLOW_KIND'] == 'production']
row_injection = dataframe[dataframe['FLOW_KIND'] == 'injection']

'''
calculate mean values of 'AVG_DOWNHOLE_TEMPERATURE' and 'AVG_DOWNHOLE_PRESS' in all production wells without zeros rows 
1. drop nan in 9 input features
2. drop zeros rows of 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'
3. drop outliers in 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'
4. calculate mean of 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'
'''
row_production = row_production.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'DP_CHOKE_SIZE', 'AVG_WHT_P', 'AVG_WHP_P', 'AVG_ANNULUS_PRESS'])

index_pressure_all = row_production.index[row_production['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
index_temp_all = row_production.index[row_production['AVG_DOWNHOLE_TEMPERATURE'] == 0].tolist()
index_unique = list(set().union(index_pressure_all, index_temp_all))
row_production = row_production.drop(index_unique)

detect_drop_outlier(row_production, ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'])

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


well_p_2 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7078) & (dataframe['FLOW_KIND'] == 'production')]
well_p_2 = well_p_2.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'DP_CHOKE_SIZE', 'AVG_WHT_P', 'AVG_WHP_P', 'AVG_ANNULUS_PRESS'])

well_p_3 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5599) & (dataframe['FLOW_KIND'] == 'production')]
well_p_3 = well_p_3.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS'])

well_p_4 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5351) & (dataframe['FLOW_KIND'] == 'production')]
well_p_4 = well_p_4.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P'])

well_p_5 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 7289) & (dataframe['FLOW_KIND'] == 'production')]

list_p = [well_p_1, well_p_2, well_p_3, well_p_4, well_p_5]

# Drop NaN row less than 5% for injection wells
well_p_6 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5693) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_6 = well_p_6.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])

well_p_7 = dataframe[(dataframe['NPD_WELL_BORE_CODE'] == 5769) & (dataframe['FLOW_KIND'] == 'Injection')]
well_p_7 = well_p_7.dropna(subset=['ON_STREAM_HRS', 'DP_CHOKE_SIZE'])

list_i = [well_p_6, well_p_7]


'''
load model and predict annulus nissing data with NN model
'''
# load json and create model
json_file = open('model_train_test_on_all.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_train_test_on_all.h5")
print("Loaded model from disk")

# predict loaded model on missing data
loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# list_t = [well_p_4]
for data in list_p:
    index_nan = data['AVG_ANNULUS_PRESS'].index[data['AVG_ANNULUS_PRESS'].apply(np.isnan)]
    index_zeros = data.index[data['AVG_ANNULUS_PRESS'] == 0].tolist()
    index_row = list(set().union(index_nan, index_zeros))
    X = data.loc[:, ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL']]
    scaler = MinMaxScaler()
    scaler.fit(X.astype(float))

    for value in index_row:
        # row_value = data.loc[value].values
        # print(row_value[12])
        input_values = data.loc[value, ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL']].values
        input_values = input_values.reshape(1, 9)
        x = scaler.transform(input_values)
        data.at[value, 'AVG_ANNULUS_PRESS'] = loaded_model.predict(x)
        # print(data.loc[value, 'AVG_ANNULUS_PRESS'])

# list_t = [well_p_1]
# for data in list_p:
#     data['DATEPRD'] = data['DATEPRD'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))

# for data in list_p:
#     fig, ax = plt.subplots()
#     ax.plot(data.DATEPRD, data.BORE_OIL_VOL)
#     fig.autofmt_xdate()
#     # format the ticks
#     datemin = datetime.date(data['DATEPRD'].min().year, data['DATEPRD'].min().month, 1)
#     datemax = datetime.date(data['DATEPRD'].max().year, data['DATEPRD'].max().month+1, 1)
#     ax.set_xlim(datemin, datemax)

## plot figures
num = 1
for data in list_p:
    features = ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE']
    # boxplot
    matplotlib.rcParams.update({'font.size': 16})
    # data[['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE']].plot(kind='box', subplots=True, layout=(4, 2), sharex=False, sharey=False)
    # plt.suptitle('Well_%i without zeros' % num)
    # # plt.savefig('Well_%i boxplot before droping outliers.png' % num)
    # # fill_data(data)
    # print(data.loc[:5, ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P',  'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL']])
    detect_drop_outlier(data, ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P',  'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE'])

    data[['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P',  'AVG_ANNULUS_PRESS','AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE']].plot(kind='box', subplots=True, layout=(4, 2), sharex=False, sharey=False)
    plt.suptitle('Well_%i imputed data' % num)
    # # plt.savefig('Well_%i boxplot after droping outliers.png' % num)

    # # correlation plot
    # matplotlib.rcParams.update({'font.size': 12})
    # names = ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL']
    # plt.figure()
    # corr = data[names].corr()
    # sns.heatmap(corr, annot=True, cbar=False)
    # matplotlib.rcParams.update({'font.size': 16})
    # plt.suptitle('Well_%i correlation plot' % num)
    # plt.savefig('Well_%i correlation plot.png' % num)

    # # pairplot
    # sns.pairplot(data, vars=['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL'])
    # plt.savefig('Well_%i pairplot.png' % num)

    num += 1


# # pairplot of all dataset
# dataset = well_p_1.append([well_p_2, well_p_3, well_p_4, well_p_5], ignore_index=True)
# sns.pairplot(dataset, vars=['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL'])
#
# for data in list_p:
#     detect_drop_outlier(data, ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL'])
#
# dataset_drop = well_p_1.append([well_p_2, well_p_3, well_p_4, well_p_5], ignore_index=True)
# sns.pairplot(dataset_drop, vars=['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE', 'BORE_OIL_VOL'])

# # PCA
#     x = data.loc[:, features].values
#     y = data.loc[:, ['BORE_OIL_VOL']].values
#     x = StandardScaler().fit_transform(x)
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(data=principalComponents
#                                , columns=['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([principalDf, data[['BORE_OIL_VOL']]], axis=1)
#     finalDf = finalDf.dropna(subset=['BORE_OIL_VOL'])
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_xlabel('Principal Component 1', fontsize=15)
#     ax.set_ylabel('Principal Component 2', fontsize=15)
#     ax.set_title('2 component PCA', fontsize=20)
#     targets = ['BORE_OIL_VOL']
#     colors = ['r', 'g', 'b']
#     for target, color in zip(targets, colors):
#         indicesToKeep = finalDf['BORE_OIL_VOL'] == target
#         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                    , finalDf.loc[indicesToKeep, 'principal component 2']
#                    , c=color
#                    , s=50)
#     ax.legend(targets)
#     ax.grid()

plt.show()



