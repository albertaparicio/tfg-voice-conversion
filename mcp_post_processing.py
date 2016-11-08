# Created by albert aparicio on 08/11/16
# coding: utf-8

# This script computes the error metrics of the GRU-RNN model for mcp mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.models import model_from_json
from keras.optimizers import RMSprop

import utils
from error_metrics import MCD

# Batch shape
batch_size = 1
tsteps = 50
data_dim = 40

# Load already trained LSTM-RNN model
print('Loading model...', end='')
with open('mcp_model.json', 'r') as model_json:
    model = model_from_json(model_json.read())

model.load_weights('mcp_weights.h5')

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)

# Load training statistics
train_stats = np.loadtxt('mcp_train_stats.csv', delimiter=',')

src_train_mean = train_stats[0, 0]
src_train_std = train_stats[0, 1]
trg_train_mean = train_stats[1, 0]
trg_train_std = train_stats[1, 1]
print('done')

# Load test data
print('Loading test data...', end='')
test_data = np.loadtxt('data/test_datatable.csv.gz', delimiter=',')

src_test_data = test_data[:, 0:40]  # Source data
src_test_data = utils.reshape_lstm(src_test_data, tsteps, data_dim)
src_test_data = (src_test_data - src_train_mean) / src_train_std

trg_test_data = test_data[:, 43:83]  # Target data
print('done')

# Predict data according to model
print('Predicting')
prediction_test = model.predict(src_test_data, batch_size=batch_size)
prediction_test = prediction_test.reshape(-1, data_dim)

# De-normalize predicted output
prediction_test = (prediction_test * trg_train_std) + trg_train_mean

# Compute RMSE of test data
mcd_test = MCD(
    trg_test_data,
    prediction_test
)

# Print resulting MCD
print('Test MCD: ', mcd_test)

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
