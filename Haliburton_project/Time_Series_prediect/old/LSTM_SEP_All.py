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
from sklearn.model_selection import train_test_split, KFold
from keras import backend
from sklearn.utils import shuffle

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


def move_oil_ToEnd(dataframe):
    out_col = dataframe.loc[:, ['BORE_OIL_VOL']]
    dataframe = dataframe.drop(["BORE_OIL_VOL"], axis=1)
    dataframe = pd.concat([dataframe, out_col], axis=1)
    return dataframe


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


wells = ['1 C', '11 H', '12 H', '14 H', '15 D']
plot_data(plot=False)


def normalization(dataset):
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset.astype('float64'))
    return dataset


def lstm_model(model_name, train=False):
    if train:
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
        # fit model
        model.fit(X_train, y_train, epochs=60, verbose=2)

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


n_steps = 10
df_NoCode = df.drop('WELL_BORE_CODE', axis=1)
df = move_oil_ToEnd(df)
df_NoCode = move_oil_ToEnd(df_NoCode)
mean, std = np.mean(df_NoCode, axis=0), np.std(df_NoCode, axis=0)
oil_std = std[-1]


def data_prepare(dataframe, name):
    df = dataframe[dataframe['WELL_BORE_CODE'] == name]
    well_code = df.loc[:, 'WELL_BORE_CODE'].values
    df = df.drop('WELL_BORE_CODE', axis=1)
    mean, std = np.mean(df, axis=0), np.std(df, axis=0)
    std = std[-1]
    mean = mean[-1]
    df = normalization(df)
    X, y = split_sequences(df, n_steps)
    well_code = well_code[:len(y)]
    print('%s shape:' % name, X.shape, y.shape, well_code.shape)
    return X, y, well_code, mean, std, df


X_1C, y_1C, well_code_1C, mean_1C, std_1C, df_1C = data_prepare(df, 'NO 15/9-F-1 C')

X_11H, y_11H, well_code_11H, mean_11H, std_11H, df_11H = data_prepare(df, 'NO 15/9-F-11 H')

X_12H, y_12H, well_code_12H, mean_12H, std_12H, df_12H = data_prepare(df, 'NO 15/9-F-12 H')

X_14H, y_14H, well_code_14H, mean_14H, std_14H, df_14H = data_prepare(df, 'NO 15/9-F-14 H')

X_15D, y_15D, well_code_15D, mean_15D, std_15D, df_15D = data_prepare(df, 'NO 15/9-F-15 D')

X = concatenate((X_1C, X_11H, X_12H, X_14H, X_15D), axis=0)
y = concatenate((y_1C, y_11H, y_12H, y_14H, y_15D), axis=0)
code = concatenate((well_code_1C, well_code_11H, well_code_12H, well_code_14H, well_code_15D), axis=0)
y_new = np.vstack([y, code])
y_new = np.transpose(y_new)
X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=seed, shuffle=True)

std = [std_1C, std_11H, std_12H, std_14H, std_15D]
std_all = std[0]*(len(df_1C)/len(X))+std[1]*(len(df_11H)/len(X))+std[2]*(len(df_12H)/len(X))+std[3]*(len(df_14H)/len(X))+std[4]*(len(df_15D)/len(X))

n_features = X.shape[2]
y_train = y_train[:, 0]
model = lstm_model('lstm_SEP10_all', train=False)
mse, r2 = model.evaluate(X_test, y_test[:, 0], verbose=2)
print('ALL: mse = ', mse, 'r2 = ', r2)

r2_list = list()
mse_list = list()
cust_list = list()


def test_model(name, mean, std):
    X = X_test[y_test[:, 1] == name]
    y = y_test[y_test[:, 1] == name][:, 0]
    mse, r2 = model.evaluate(X, y, verbose=2)
    y_pred = model.predict(X)
    cust = np.mean(np.abs(y_pred - y.reshape(len(y), 1)) / (
                y.reshape(len(y), 1) + mean / std))
    print('1C: mse = ', mse, 'r2 = ', r2, 'cust = ', cust)
    r2_list.append(r2)
    mse_list.append(mse)
    cust_list.append(cust)


# test on 1C
test_model('NO 15/9-F-1 C', mean_1C, std_1C)

# test on 11H
test_model('NO 15/9-F-11 H', mean_11H, std_11H)

# test on 12H
test_model('NO 15/9-F-12 H', mean_12H, std_12H)

# test on 14H
test_model('NO 15/9-F-14 H', mean_14H, std_14H)

# test on 15D
test_model('NO 15/9-F-15 D', mean_15D, std_15D)

print('mse: ', np.mean(mse_list), 'r2:', np.mean(r2_list), 'cust', np.mean(cust_list))

# K fold
# kf = KFold(n_splits=2, random_state=None, shuffle=False)
# for j, (train_index, test_index) in enumerate(kf.split(X_s, y_s)):
#     # define model
#     X_train, X_test = X_s[train_index], X_s[test_index]
#     y_train, y_test = y_s[train_index], y_s[test_index]
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
#     # fit model
#     model.fit(X_train, y_train, epochs=60, verbose=2, validation_data=(X_test, y_test))

# define model