import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

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
def missiong_ratio(data):
    data_missing = data.isna()
    num_missing = data_missing.sum()
    percentage = num_missing/len(data)
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
# missiong_ratio(production_over_thre)
# drop nan less than 5% of production > 10
production_over_thre = production_over_thre.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_WHT_P', 'AVG_WHP_P'])

df = production_over_thre[(dataframe['NPD_WELL_BORE_CODE'] == 5599) | (dataframe['NPD_WELL_BORE_CODE'] == 5351)]
dataset = df.dropna(subset=['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P', 'AVG_WHP_P'])
# missiong_ratio(dataset)
index_0 = dataset.index[dataset['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
dataset = dataset.drop(index_0)
detect_drop_outlier(dataset, ['AVG_DOWNHOLE_PRESSURE'])
dataset_value = dataset.values
X = dataset_value[:, [8, 11, 13, 14, 15]]
Y = dataset_value[:, 9]
Y = np.reshape(Y, (-1, 1))

scaler = MinMaxScaler()
print(scaler.fit(X.astype(float)))
Xscale = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xscale, Y, random_state=seed)
model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=[r2_keras])
history = model.fit(X_train, y_train, epochs=500, batch_size=50,  verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test)
print(model.metrics_names)
print('mse: %.2f' % (score[0]))
print('r2: %.2f%%' % (score[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model_MLP_PRESS.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_MLP_PRESS.h5")
print("Saved model to disk")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
