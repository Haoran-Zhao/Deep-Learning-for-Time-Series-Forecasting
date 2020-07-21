import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from pandas import concat
from numpy import array
import numpy as np
from numpy import concatenate
from sklearn.model_selection import train_test_split
from keras import backend
from sklearn.utils import shuffle

# normalize to each well, each well provide 80% for train set
seed = 7
np.random.seed(seed)

# Load data
dataset = pd.read_excel('data_ok.xlsx', header=0)
dataset['DATEPRD'] = pd.to_datetime(dataset["DATEPRD"]).dt.date
# dataset.drop('WELL_BORE_CODE', axis=1, inplace=True)

# Manually specify cols names
df = dataset.set_index('DATEPRD')
print(df.keys())


# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())


def plot_data(plot=False):
    if plot==True:
        # specify groups to plot
        groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # plot each cols
        for i in range(1, len(wells)+1):
            j = 1
            plt.figure(i)
            data_plot = df[df['WELL_BORE_CODE'] == 'NO 15/9-F-%s' % wells[i-1]]
            values = data_plot.values
            for group in groups:
                plt.subplot(len(groups), 1, j)
                plt.plot(values[:, group])
                plt.title(df.columns[group], y=0.5, loc='right')
                plt.suptitle('%s' % wells[i-1])
                # plt.xticks(range(0, dataset.shape[0], 500), dataset['DATEPRD'].loc[::500], rotation=45)
                j += 1

        plt.figure(i+1)
        plt.scatter(range(dataset.shape[0]), dataset['BORE_OIL_VOL'])
        plt.xticks(range(0, dataset.shape[0], 500), dataset['DATEPRD'].loc[::500], rotation=45)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Bore_Oil_Vol', fontsize=18)

        plt.show()


wells = ['1 C', '11 H', '12 H', '14 H', '15 D']
plot_data(plot=False)


def move_oil_ToEnd(dataframe):
    out_col = dataframe.loc[:, ['BORE_OIL_VOL']]
    dataframe = dataframe.drop(["BORE_OIL_VOL"], axis=1)
    dataframe = pd.concat([dataframe, out_col], axis=1)
    return dataframe


def lstm_model(model_name, train=False):
    if train:
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
        # fit model
        model.fit(X_train, y_train[:, 0], epochs=60, verbose=0)

        # Save model
        model_json = model.to_json()
        with open("%s_model.json" % model_name, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("%s_model.h5" % model_name)
        print("Saving %s model to disk .." % model_name)
    else:
        json_file = open('%s_model.json' % model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('%s_model.h5' % model_name)
        print('loaded model from disk')

        model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])

    return model


def test_model(name):
    X = X_test[y_test[:, 1] == name]
    y = y_test[y_test[:, 1] == name][:, 0]
    mse0, r20 = model.evaluate(X, y, verbose=0)
    # y_pred = model.predict(X)
    # cust0 = np.mean(np.abs(y_pred - y.reshape(len(y), 1)) / (
    #             y.reshape(len(y), 1) + mean / std))
    print('%s : mse = ' % name, mse0, 'r2 = ', r20)
    r2_list.append(r2)
    mse_list.append(mse)
    # cust_list.append(cust0)


def fintune(name, model, model_name, train=False, epoch=10):
    X_trainset = X_train[y_train[:, 1] == name]
    y_trainset = y_train[y_train[:, 1] == name][:, 0]
    X = X_test[y_test[:, 1] == name]
    y = y_test[y_test[:, 1] == name][:, 0]
    if train:
        model.fit(X_trainset, y_trainset, epochs=epoch, verbose=0)
        mse0, r20 = model.evaluate(X, y, verbose=0)
        # y_pred = model.predict(X)
        # cust0 = np.mean(np.abs(y_pred - y.reshape(len(y), 1)) / (
        #             y.reshape(len(y), 1) + mean / std))
        print('%s : mse = ' % name, mse0, 'r2 = ', r20)
        r2_list.append(r2)
        mse_list.append(mse)
        # cust_list.append(cust0)
        # Save model
        model_json = model.to_json()
        with open("%s_model.json" % model_name, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("%s_model.h5" % model_name)
        print("Saving %s model to disk .." % model_name)
    else:
        json_file = open('%s_model.json' % model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('%s_model.h5' % model_name)
        print('loaded model from disk')
        model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
        mse0, r20 = model.evaluate(X, y, verbose=0)
        # y_pred = model.predict(X)
        # cust0 = np.mean(np.abs(y_pred - y.reshape(len(y), 1)) / (
        #             y.reshape(len(y), 1) + mean / std))
        print('%s : mse = ' % name, mse0, 'r2 = ', r20)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def split_well(dataframe, name):
    dataframe_split = dataframe[dataframe['WELL_BORE_CODE'] == name]
    well_code = dataframe_split.loc[:, 'WELL_BORE_CODE'].values
    dataframe_split = dataframe_split.drop('WELL_BORE_CODE', axis=1)
    return dataframe_split, well_code


def data_prepare(dataframe, name, trainset_ratio):
    df0, well_code = split_well(dataframe, name)
    X, y = split_sequences(df0.values, n_steps)
    y = y.reshape(-1, 1)
    well_code = well_code.reshape(-1, 1)
    y = np.hstack([y, well_code[:len(y)]])
    X, y = shuffle(X, y)
    X_train0, X_test0, y_train0, y_test0 = X[int(trainset_ratio * len(df0)):, :], X[:int(trainset_ratio * len(df0)), :], y[int(trainset_ratio * len(df0)):], y[:int(trainset_ratio * len(df0))]
    mean, std = np.mean(y_train0[:, 0], axis=0), np.std(y_train0[:, 0], axis=0)

    mean_x, std_x = np.mean(X_train0, axis=0), np.std(X_train0, axis=0)
    train_X0 = (X_train0-mean_x)/std_x
    test_X0 = (X_test0-mean_x)/std_x

    train_y0 = y_train0
    test_y0 = y_test0

    train_y0[:, 0] = (y_train0[:, 0]-mean)/std
    test_y0[:, 0] = (y_test0[:, 0] - mean)/std

    return train_X0, test_X0, train_y0, test_y0, mean, std


n_steps = 10
df_NoCode = df.drop('WELL_BORE_CODE', axis=1)
df = move_oil_ToEnd(df)
df_NoCode = move_oil_ToEnd(df_NoCode)

X_1C_train, X_1C_test, y_1C_train, y_1C_test, mean_1C, std_1C = data_prepare(df, 'NO 15/9-F-1 C', 0.2)
X_11H_train, X_11H_test, y_11H_train, y_11H_test, mean_11H, std_11H = data_prepare(df, 'NO 15/9-F-11 H', 0.2)

X_train = concatenate((X_1C_train, X_11H_train), axis=0)
y_train = concatenate((y_1C_train, y_11H_train), axis=0)

X_test = concatenate((X_1C_test, X_11H_test), axis=0)
y_test = concatenate((y_1C_test, y_11H_test), axis=0)

for i in range(1, 11):
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    n_features = X_train.shape[2]
    model = lstm_model('lstm_1C_11H_EACH', train=True)
    mse, r2 = model.evaluate(X_test, y_test[:, 0], verbose=0)
    print('ALL: mse = ', mse, 'r2 = ', r2)

    r2_list = list()
    mse_list = list()
    # cust_list = list()

    # test on 1C
    test_model('NO 15/9-F-1 C')

    # test on 11H
    test_model('NO 15/9-F-11 H')


# model_1C = model
# model_11H = model
#
#
#
# # fine tune on each well
# fintune('NO 15/9-F-1 C', model_1C, 'lstm_EACH_EACH_1C', train=False, epoch=0)
#
# fintune('NO 15/9-F-11 H', model_11H, 'lstm_EACH_EACH_11H', train=False, epoch=60)




