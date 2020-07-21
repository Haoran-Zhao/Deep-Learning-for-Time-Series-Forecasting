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
dataframe = pd.read_excel("overall_mean_data.xlsx", sep='delimiter', header=0)
print(dataframe.keys())


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


dataset = dataframe.copy()
# missiong_ratio(dataset)
index_0 = dataset.index[dataset['AVG_DOWNHOLE_PRESSURE'] == 0].tolist()
index_nan = dataset['AVG_ANNULUS_PRESS'].index[dataset['AVG_ANNULUS_PRESS'].apply(np.isnan)].tolist()
dataset = dataset.drop(index_0)
dataset = dataset.drop(index_nan)
dataset_value = dataset.values
X = dataset_value[:, [8, 9, 10, 11, 13, 14, 15, 16]]
Y = dataset_value[:, 12]
Y = np.reshape(Y, (-1, 1))

scaler = MinMaxScaler()
print(scaler.fit(X.astype(float)))
Xscale = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xscale, Y, random_state=seed)
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='normal', activation='relu'))
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

# # serialize model to JSON
# model_json = model.to_json()
# with open("model_MLP_PRESS.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_MLP_PRESS.h5")
# print("Saved model to disk")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
