import sys, os, math
dir =  os.getcwd()

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import pickle
# Multilayer Perceptron Regression
# ====================================
import matplotlib.pyplot as plt
import numpy as np, pandas
import seaborn as sns
# fix random seed for reproducibility
np.random.seed(7)

monthDict = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06", "Jul":"07",
            "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}

rawdata = pandas.ExcelFile(os.path.join(dir, 'datasets_wi.xlsx'))

vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL', 'BORE_WI_VOL','DATEPRD']

# Convert dates to integer
label = 'DATEPRD'
sheet_name = rawdata.sheet_names
sheet = rawdata.parse(sheet_name=sheet_name[0])
rawdates = pandas.to_datetime(sheet[label].values)

for id, it in enumerate(rawdates):
  sheet[label].values[id] = 10000*rawdates.year[id] + 100*rawdates.month[id] + rawdates.day[id]

dates = np.array(sheet[label].values, dtype=np.int64)
uniq_t = np.sort(np.unique(dates))
t_dict = {i:j for i,j in zip(uniq_t, np.arange(np.shape(uniq_t)[0]) )}

irrelevant_features = ['DATEPRD', 'WELL_BORE_CODE',
                       'FLOW_KIND', 'BORE_GAS_VOL',
                       'BORE_WAT_VOL']

relevant_features = ['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DOWNHOLE_PRESSURE', 'AVG_ANNULUS_PRESS',
                     'ON_STREAM_HRS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
                     'AVG_WHT_P', 'DP_CHOKE_SIZE', 'WI1', 'WI2',
                     'BORE_OIL_VOL']

num_features = len(relevant_features)
data_size = len(sheet[relevant_features[0]].values)
dataset = np.empty((data_size, num_features))
for idx, feature in enumerate(relevant_features):
  dataset[:, idx] = np.array(sheet[feature].values, dtype=np.float32)



# Normalization
mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)
print(std)
norm_set = (dataset - mean) / (std + 1e-6)

# Convert normalized data to sequences by wells
sequences = []
well_code_col = sheet['WELL_BORE_CODE'].values
uniq_well_code = np.unique(well_code_col)
for id, well in enumerate(uniq_well_code):
  sequences.append(norm_set[well_code_col == well, :])

# split a multi-variate sequence into samples
def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
      # find the end of this pattern
      end_ix = i + n_steps
      # check if we are beyond the sequence
      if end_ix > len(sequence):
          break
      # gather input and output parts of the pattern
      seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)

from sklearn.model_selection import KFold
from keras.models import Sequential, model_from_json
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K

def r2(label, pred):
  SS_res = K.sum(K.square(label - pred ))
  SS_tot = K.sum(K.square(label - K.mean(label)))
  return (1 - SS_res/ (SS_tot + K.epsilon()))

# define model
def get_model(n_steps):
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(n_steps, num_features - 1)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=[r2])
  return model

n_steps = 10

def train_test_split(sequences, n_steps):

  train_X = np.empty((0, n_steps, num_features-1))
  train_y = np.empty((0))

  for sequence in sequences:
    X, y = split_sequence(sequence=sequence, n_steps=n_steps)
    train_X = np.append(train_X, np.array(X), axis=0)
    train_y = np.append(train_y, np.array(y), axis=0)

  train_X, train_y = shuffle(train_X, train_y)

  train_X, train_y, test_X, test_y = \
    train_X[int(0.2*data_size):, :], train_y[int(0.2*data_size):], \
    train_X[:int(0.2*data_size), :], train_y[:int(0.2*data_size)]

  return train_X, train_y, test_X, test_y

def lstm_model(model_name, n_steps, X, y, cross_val=False, nfolds=10, run_train=False):

  if not run_train and os.path.isfile(os.path.join(dir, 'models', '%s_model.json' % (model_name))):
    # Load model
    f = open(os.path.join(dir, 'models', '%s_model.json' % (model_name)), 'r')
    loaded_json = f.read()
    f.close()
    model = model_from_json(loaded_json)
    model.load_weights(os.path.join(dir, 'models', '%s_model.h5' % (model_name)))
    print("Model loaded from %s_model.h5" % (model_name))
    model.compile(optimizer='adam', loss='mse', metrics=[r2])
  else:
    if cross_val:
      folds = list(KFold(n_splits=nfolds, shuffle=False, random_state=None).split(X, y))
      testScore = []
      for j, (train_idx, val_idx) in enumerate(folds):
        train_X, train_y = X[train_idx, :], y[train_idx]
        valid_X, valid_y = X[val_idx, :], y[val_idx]
        model = get_model(n_steps)
        res = model.fit(train_X, train_y, validation_split=0.01, epochs=60, verbose=0)
        _, Score = model.evaluate(valid_X, valid_y, verbose=0)
        print('Fold ',j, 'cv score = ', Score)
        testScore.append(Score)
      print('\n', 'n_steps = ', n_steps, 'cv r2 score = ', np.mean(testScore))

    model = get_model(n_steps)
    model.fit(X, y, epochs=60, verbose=2)

    # Save model
    model_json = model.to_json()
    with open(os.path.join(dir, 'models', "%s_model.json" % (model_name)), "w") as f:
      f.write(model_json)
    model.save_weights(os.path.join(dir, 'models', "%s_model.h5" % (model_name)))
    print("Saving %s model to disk .." % (model_name))

  return model


train_X, train_y, test_X, test_y = train_test_split(sequences, n_steps)

# set run_train = True to train the model
# set cross_val = True to enable cross validation
model = lstm_model(model_name='lstm10wi', X=train_X, y=train_y,
                   n_steps=n_steps, cross_val=False, run_train=False)

mse, r2 = model.evaluate(test_X, test_y, verbose=0)
print('mse = ', mse, 'r2 = ', r2)
