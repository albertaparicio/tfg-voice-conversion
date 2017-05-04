# Created by albert aparicio on 08/11/16
# coding: utf-8

# This script computes the error metrics of the GRU-RNN model for mcp mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os

import h5py
import matplotlib
from ahoproc_tools.error_metrics import MCD
from keras.models import model_from_json
from keras.optimizers import RMSprop
from tfglib import utils

matplotlib.use('TKagg')
from matplotlib import pyplot as plt

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 40

##############
# Load model #
##############
# Load already trained LSTM-RNN model
print('Loading model...', end='')
with open('models/mcp_model.json', 'r') as model_json:
  model = model_from_json(model_json.read())

model.load_weights('models/mcp_weights.h5')

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)
print('done')

#############
# Load data #
#############
# Load training statistics
with h5py.File('models/mcp_train_stats.h5', 'r') as train_stats:
  src_train_mean = train_stats['src_train_mean'][:]
  src_train_std = train_stats['src_train_std'][:]
  trg_train_mean = train_stats['trg_train_mean'][:]
  trg_train_std = train_stats['trg_train_std'][:]

  train_stats.close()

# Load test data
print('Loading test data...', end='')
with h5py.File('data/test_datatable.h5', 'r') as test_datatable:
  test_data = test_datatable['test_data'][:, :]

  test_datatable.close()

src_test_data = test_data[:, 0:40]  # Source data
src_test_data = utils.reshape_lstm(src_test_data, tsteps, data_dim)
src_test_data = (src_test_data - src_train_mean) / src_train_std

trg_test_data = test_data[:, 43:83]  # Target data
print('done')

################
# Predict data #
################
print('Predicting')
prediction_test = model.predict(src_test_data, batch_size=batch_size)
prediction_test = prediction_test.reshape(-1, data_dim)

# De-normalize predicted output
prediction_test = (prediction_test * trg_train_std) + trg_train_mean

#################
# Error metrics #
#################
# Compute MCD of test data
mcd_test = MCD(
    trg_test_data,
    prediction_test
    )

# Print resulting MCD
print('Test MCD: ', mcd_test)

# Load training parameters and save loss curves
with h5py.File('training_results/baseline/mcp_history.h5', 'r') as hist_file:
  loss = hist_file['loss'][:]
  val_loss = hist_file['val_loss'][:]
  epoch = hist_file['epoch'][:]

  hist_file.close()

print('Saving loss curves')

plt.plot(epoch, loss, epoch, val_loss)
plt.legend(['loss', 'val_loss'], loc='best')
plt.grid(b=True)
plt.suptitle('Baseline MVF Loss curves')
plt.savefig(os.path.join('training_results', 'baseline', 'mcp_loss_curves.eps'),
            bbox_inches='tight')

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

exit()
