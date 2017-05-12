# Created by albert aparicio on 04/11/16
# coding: utf-8

# This script computes the error metrics of the LSTM-RNN model for lf0 mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os

import h5py
import matplotlib
import numpy as np
from ahoproc_tools.error_metrics import AFPR, RMSE
from keras.models import model_from_json
from scipy.stats import pearsonr
from tfglib import utils

matplotlib.use('TKagg')
from matplotlib import pyplot as plt

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 2

##############
# Load model #
##############
# Load already trained LSTM-RNN model
with open('models/lf0_model.json', 'r') as model_json:
  model = model_from_json(model_json.read())

model.load_weights('models/lf0_weights.h5')
model.compile(loss='mse', optimizer='rmsprop')

#############
# Load data #
#############
# Load training statistics
with h5py.File('models/lf0_train_stats.h5', 'r') as train_stats:
  src_train_mean = train_stats['src_train_mean'].value
  src_train_std = train_stats['src_train_std'].value
  trg_train_mean = train_stats['trg_train_mean'].value
  trg_train_std = train_stats['trg_train_std'].value

  train_stats.close()

# Load test data
print('Loading test data...', end='')
with h5py.File('data/test_datatable.h5', 'r') as test_datatable:
  test_data = test_datatable['test_data'][:, :]

  test_datatable.close()

src_test_frames = np.column_stack((test_data[:, 40], test_data[:, 42]))
src_test_frames[:, 0] = (src_test_frames[:, 0] - src_train_mean) / src_train_std
src_test_frames = utils.reshape_lstm(src_test_frames, tsteps, data_dim)

# Zero-pad and reshape target test data
trg_test_frames = utils.reshape_lstm(
    np.column_stack((test_data[:, 83], test_data[:, 85])), tsteps,
    data_dim).reshape(-1, 2)

print('done')

################
# Predict data #
################
print('Predicting')
prediction_test = model.predict(src_test_frames, batch_size=batch_size)
prediction_test = prediction_test.reshape(-1, 2)

# De-normalize predicted output
prediction_test[:, 0] = (prediction_test[:, 0] * trg_train_std) + trg_train_mean

#################
# Error metrics #
#################
# Compute and print RMSE of test data
rmse_test = RMSE(
    np.exp(trg_test_frames[:, 0]),
    np.exp(prediction_test[:, 0]),
    mask=trg_test_frames[:, 1]
    )

print('Test RMSE: ', rmse_test)

# Compute and print Pearson correlation between target and prediction
pearson = pearsonr(
    np.exp(trg_test_frames[:, 0]),
    np.exp(prediction_test[:, 0])
    )

print('Pearson correlation: ', pearson[0])

# Round the predicted flags to get binary values
prediction_test[:, 1] = np.round(prediction_test[:, 1])

# Compute Accuracy of U/V flag prediction
accuracy, _, _, _ = AFPR(trg_test_frames[:, 1], prediction_test[:, 1])

print('U/V Accuracy: {}%'.format(accuracy * 100))
# print('F-measure: ', accuracy[1] * 100, '%')
# print('Precision: ', accuracy[2] * 100, '%')
# print('Recall: ', accuracy[3] * 100, '%')

# Load training parameters and save loss curves
with h5py.File('training_results/baseline/lf0_history.h5', 'r') as hist_file:
  loss = hist_file['loss'][:]
  val_loss = hist_file['val_loss'][:]
  epoch = hist_file['epoch'][:]

  hist_file.close()

print('Saving loss curves')

plt.plot(epoch, loss, epoch, val_loss)
plt.legend(['loss', 'val_loss'], loc='best')
plt.grid(b=True)
plt.suptitle('Baseline log(f0) Loss curves')
plt.savefig(os.path.join('training_results', 'baseline', 'lf0_loss_curves.eps'),
            bbox_inches='tight')

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

exit()
