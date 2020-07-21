import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

seed = 7
np.random.seed(seed)


def split_dataframe(dataframe):
    train_set = dataframe[dataframe['class'] > 2].values
    test_set = dataframe[(dataframe['class'] < 3) & (dataframe['class'] > 0)].values
    return train_set, test_set


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


df = pd.read_excel("op_wi.xlsx", sep='delimiter', header=0)
df['DATEPRD'] = pd.to_datetime(df["DATEPRD"]).dt.date
print(df.head())
print(df.keys())

# plt.figure(figsize=(18, 9))
# plt.scatter(range(df.shape[0]), df['BORE_OIL_VOL'])
# plt.xticks(range(0, df.shape[0], 500), df['DATEPRD'].loc[::500], rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Bore_Oil_Vol', fontsize=18)
# plt.show()

dataset = df.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

reframed = series_to_supervised(dataset_scaled, 1, 1)

