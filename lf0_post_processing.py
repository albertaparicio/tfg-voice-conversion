# Created by albert aparicio on 04/11/16
# coding: utf-8

# This script computes the error metrics of the LSTM-RNN model for log(f0) mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.models import load_model

import utils
from error_metrics import RMSE

# Batch shape
batch_size = 1
tsteps = 50
data_dim = 2

# Load already trained LSTM-RNN model
model = load_model('lfo_lstm_model.h5')

# Load training statistics
train_stats = np.loadtxt('lf0_train_stats.csv', delimiter=',')

src_train_mean = train_stats[0, 0]
src_train_std = train_stats[0, 1]
trg_train_mean = train_stats[1, 0]
trg_train_std = train_stats[1, 1]

print('Loading test datatable...', end='')
test_data = np.loadtxt('data/test_datatable.csv.gz', delimiter=',')
print('done')

src_test_frames = np.column_stack((test_data[:, 40], test_data[:, 42]))
src_test_frames[:, 0] = (src_test_frames[:, 0] - src_train_mean) / src_train_std
src_test_frames = utils.reshape_lstm(src_test_frames, tsteps, data_dim)

trg_test_frames = np.column_stack((test_data[:, 83], test_data[:, 85]))
trg_test_frames = utils.reshape_lstm(trg_test_frames, tsteps, data_dim)

print('Predicting')
prediction_test = model.predict(src_test_frames, batch_size=batch_size)

# De-normalize predicted output
prediction_test[:, 0] = (prediction_test[:, 0] * trg_train_std) + trg_train_mean

# Compute RMSE of test data
rmse_test = RMSE(np.exp(trg_test_frames[:, 0]), np.exp(prediction_test[:, 0]), mask=trg_test_frames[:, 1])

# Print resulting RMSE
print('Test RMSE: ', rmse_test)
