import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras import backend
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)


# r squared
def r2_keras(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/(SS_tot + backend.epsilon())


# data load
dataframe = pd.read_excel("cleaned_imputed_Oil_production.xlsx", sep='delimiter', header=0)
print(dataframe.keys())

dataset = dataframe.values
X = dataset[:, 2:11]
Y = dataset[:, 11]

scaler = MinMaxScaler()
scaler.fit(X.astype(float))
X_scale = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=9, kernel_initializer="normal", activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=[r2_keras])
history = model.fit(X_train, y_train, epochs=500, batch_size=50,  verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test)
print(model.metrics_names)
print('mse: %.2f' % (score[0]))
print('r2: %.2f%%' % (score[1]*100))