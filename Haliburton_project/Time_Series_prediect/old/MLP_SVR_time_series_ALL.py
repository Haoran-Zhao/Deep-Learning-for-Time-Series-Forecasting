import pandas as pd
import numpy as np
from pandas import concat
from numpy import concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import Conv1D, Flatten
from keras.layers import MaxPool1D
from sklearn import svm, metrics
from sklearn.utils import shuffle

# normalize to all 0.8*data for train set

seed = 7
np.random.seed(seed)
wells = ['1 C', '11 H', '12 H', '14 H', '15 D']

# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())


def move_oil_ToEnd(dataframe):
    out_col = dataframe.loc[:, ['BORE_OIL_VOL']]
    dataframe = dataframe.drop(["BORE_OIL_VOL"], axis=1)
    dataframe = pd.concat([dataframe, out_col], axis=1)
    return dataframe


def series_to_supervised(dataframe, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(dataframe) is list else dataframe.shape[1]
    dataframe = pd.DataFrame(dataframe)
    cols, names = list(), list()
    # input sequence(t-n,..t-1)
    for i in range(n_in, 0, -1):
        cols.append(dataframe.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
        names += [('Output(t-%d)' % i)]

    # forcast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(dataframe.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars-1)]
            names += ['Output(t)']
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            names += ['Output(t)']

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with Nan
    if dropnan:
        agg.dropna(inplace=True)
    return agg


df = pd.read_excel("data_ok.xlsx", sep='delimiter', header=0)
df['DATEPRD'] = pd.to_datetime(df["DATEPRD"]).dt.date
# print(df.head())
# print(df.keys())
df = move_oil_ToEnd(df)
df = df.set_index('DATEPRD')
print(df.keys())
# plt.figure(figsize=(18, 9))
# plt.scatter(range(df_12H.shape[0]), df_12H['BORE_OIL_VOL'])
# plt.xticks(range(0, df_12H.shape[0], 500), df_12H['DATEPRD'].loc[::500], rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Bore_Oil_Vol', fontsize=18)
# plt.show()


def split_well(dataframe, name):
    dataframe_split = dataframe[dataframe['WELL_BORE_CODE'] == name]
    well_code = dataframe_split.loc[:, 'WELL_BORE_CODE'].values
    dataframe_split = dataframe_split.drop('WELL_BORE_CODE', axis=1)
    return dataframe_split, well_code


def split_train_test(data, well_code, ratio):
    y = data.values[:, -1]
    X = data.values[:, :-1]
    well_code = well_code.reshape(-1, 1)
    y = y.reshape(-1, 1)
    y = np.hstack([y, well_code[:len(y)]])
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = X[int(ratio * len(X)):, :], X[:int(0.2 * len(X)), :], y[int(0.2 * len(X)):], y[:int(0.2 * len(X))]
    return X_train, X_test, y_train, y_test


def data_prepare(dataframe, ratio):
    df0, well_code0 = split_well(dataframe, 'NO 15/9-F-1 C')
    df1, well_code1 = split_well(dataframe, 'NO 15/9-F-11 H')
    df2, well_code2 = split_well(dataframe, 'NO 15/9-F-12 H')
    df3, well_code3 = split_well(dataframe, 'NO 15/9-F-14 H')
    df4, well_code4 = split_well(dataframe, 'NO 15/9-F-15 D')

    data0 = series_to_supervised(df0.values, n_steps)
    data1 = series_to_supervised(df1.values, n_steps)
    data2 = series_to_supervised(df2.values, n_steps)
    data3 = series_to_supervised(df3.values, n_steps)
    data4 = series_to_supervised(df4.values, n_steps)

    X_train0, X_test0, y_train0, y_test0 = split_train_test(data0, well_code0, ratio)
    X_train1, X_test1, y_train1, y_test1 = split_train_test(data1, well_code1, ratio)
    X_train2, X_test2, y_train2, y_test2 = split_train_test(data2, well_code2, ratio)
    X_train3, X_test3, y_train3, y_test3 = split_train_test(data3, well_code3, ratio)
    X_train4, X_test4, y_train4, y_test4 = split_train_test(data4, well_code4, ratio)

    X_train = concatenate((X_train0, X_train1, X_train2, X_train3, X_train4), axis=0)
    X_test = concatenate((X_test0, X_test1, X_test2, X_test3, X_test4), axis=0)
    mean_x, std_x = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    train_X = (X_train - mean_x) / std_x
    test_X = (X_test - mean_x) / std_x

    y_train = concatenate((y_train0, y_train1, y_train2, y_train3, y_train4), axis=0)
    y_test = concatenate((y_test0, y_test1, y_test2, y_test3, y_test4), axis=0)
    mean, std = np.mean(y_train[:, 0], axis=0), np.std(y_train[:, 0], axis=0)
    train_y = y_train
    test_y = y_test

    train_y[:, 0] = (y_train[:, 0]-mean)/std
    test_y[:, 0] = (y_test[:, 0] - mean)/std
    return train_X, test_X, train_y, test_y, mean, std


n_steps = 1
X_train, X_test, y_train, y_test, mean, std = data_prepare(df, 0.2)


def test_dataset_MLP(name):
    X = X_test[y_test[:, 1] == name]
    y = y_test[y_test[:, 1] == name][:, 0]
    y_pred = model.predict(X)
    scores = model.evaluate(X, y, verbose=0)
    cust = np.mean(np.abs(y_pred-y.reshape(len(y), 1))/(y.reshape(len(y), 1)+mean/std))
    scores.append(cust)
    print("MLP %s: " % name, scores)


input_size = 12*(n_steps+1)-1
# create and fit MLP
model = Sequential()
model.add(Dense(input_size, input_dim=input_size, activation='relu'))
model.add(Dense(10, activation='relu'))
model. add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=[r2_keras])
model.fit(X_train, y_train[:, 0], epochs=60, verbose=2)


scores = model.evaluate(X_test, y_test[:, 0], verbose=0)
print('MLP: ', scores)

# test on 1C
test_dataset_MLP('NO 15/9-F-1 C')

# test on 11H
test_dataset_MLP('NO 15/9-F-11 H')

# test on 12H
test_dataset_MLP('NO 15/9-F-12 H')

# test on 14H
test_dataset_MLP('NO 15/9-F-14 H')

# test on 15D
test_dataset_MLP('NO 15/9-F-15 D')


clf = svm.SVR(kernel='rbf', gamma=0.02, C=9)
clf.fit(X_train, y_train[:, 0])
y_pred = clf.predict(X_test)
print("SVR gamma = 0.02 C=9:", metrics.r2_score(y_test[:, 0], y_pred))


def test_dataset_SVR(name):
    X = X_test[y_test[:, 1] == name]
    y = y_test[y_test[:, 1] == name][:, 0]
    y_pred = clf.predict(X)
    cust = np.mean(np.abs(y_pred-y.reshape(len(y), 1))/(y.reshape(len(y), 1)+mean/std))
    print("SVR %s: " % name, [metrics.mean_squared_error(y, y_pred), metrics.r2_score(y, y_pred), cust])


# test on 1C
test_dataset_SVR('NO 15/9-F-1 C')

# test on 11H
test_dataset_SVR('NO 15/9-F-11 H')

# test on 12H
test_dataset_SVR('NO 15/9-F-12 H')

# test on 14H
test_dataset_SVR('NO 15/9-F-14 H')

# test on 15D
test_dataset_SVR('NO 15/9-F-15 D')

# cnn
# train_X_cnn = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(23, 1)))
# model.add(MaxPool1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
#
# model.fit(train_X_cnn, train_Y, epochs=200, verbose=2, validation_data=(test_X_cnn, test_Y))

# # lstm
# # define model
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(23, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
# # fit model
# model.fit(train_X_cnn, train_Y, epochs=200, verbose=2, validation_data=(test_X_cnn, test_Y))