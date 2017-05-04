# Created by albert aparicio on 06/11/16
# coding: utf-8

# This script defines a GRU-RNN to map the cepstral components of the signal

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os

import h5py
import numpy as np
from keras.layers import Dense, Dropout, GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop
from tfglib import construct_table as ct, utils

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 40

# Other constants
epochs = 50
# epochs = 25

#############
# Load data #
#############
#  Switch to decide if datatable must be build or can be loaded from a file
build_datatable = False

print('Starting...')

if build_datatable:
  # Build datatable of training and test data
  # (data is already encoded with Ahocoder)
  print('Saving training datatable...', end='')
  train_data = ct.save_datatable(
      'data/training/',
      'train_data',
      'data/train_datatable'
      )
  print('done')

  print('Saving test datatable...', end='')
  test_data = ct.save_datatable(
      'data/test/',
      'test_data',
      'data/test_datatable'
      )
  print('done')

else:
  # Retrieve datatables from .h5 files
  print('Loading training datatable...', end='')
  train_data = ct.load_datatable(
      'data/train_datatable.h5',
      'train_data'
      )
  print('done')

  print('Loading test datatable...', end='')
  test_data = ct.load_datatable(
      'data/test_datatable.h5',
      'test_data'
      )
  print('done')

################
# Prepare data #
################
# Take MCP parameter columns
src_train_data = train_data[0:17500, 0:40]  # Source data
trg_train_data = train_data[0:17500, 43:83]  # Target data

src_valid_data = train_data[17500:train_data.shape[0], 0:40]  # Source data
trg_valid_data = train_data[17500:train_data.shape[0], 43:83]  # Target data

src_test_data = test_data[:, 0:40]  # Source data
trg_test_data = test_data[:, 43:83]  # Target data

# Remove means and normalize
src_train_mean = np.mean(src_train_data, axis=0)
src_train_std = np.std(src_train_data, axis=0)

src_train_data = (src_train_data - src_train_mean) / src_train_std
src_valid_data = (src_valid_data - src_train_mean) / src_train_std
src_test_data = (src_test_data - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_data, axis=0)
trg_train_std = np.std(trg_train_data, axis=0)

trg_train_data = (trg_train_data - trg_train_mean) / trg_train_std
trg_valid_data = (trg_valid_data - trg_train_mean) / trg_train_std

# Zero-pad and reshape data
src_train_data = utils.reshape_lstm(src_train_data, tsteps, data_dim)
src_valid_data = utils.reshape_lstm(src_valid_data, tsteps, data_dim)
src_test_data = utils.reshape_lstm(src_test_data, tsteps, data_dim)

trg_train_data = utils.reshape_lstm(trg_train_data, tsteps, data_dim)
trg_valid_data = utils.reshape_lstm(trg_valid_data, tsteps, data_dim)

# Save training statistics
with h5py.File('models/mcp_train_stats.h5', 'w') as f:
  h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
  h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
  h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
  h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

  f.close()

################
# Define Model #
################
# Define an GRU-based RNN
print('Creating Model')
model = Sequential()

model.add(GRU(units=70,
              batch_input_shape=(batch_size, tsteps, data_dim),
              return_sequences=True,
              stateful=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(data_dim)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)

###############
# Train model #
###############
print('Training')
epoch = list(range(epochs))
loss = []
val_loss = []

for i in range(epochs):
  print('Epoch', i, '/', epochs)
  history = model.fit(src_train_data,
                      trg_train_data,
                      batch_size=batch_size,
                      verbose=1,
                      epochs=1,
                      shuffle=False,
                      validation_data=(src_valid_data, trg_valid_data))

  loss.append(history.history['loss'])
  val_loss.append(history.history['val_loss'])

  model.reset_states()

print('Saving model')
model.save_weights('models/mcp_weights.h5')

with open('models/mcp_model.json', 'w') as model_json:
  model_json.write(model.to_json())

print('Saving training results')
with h5py.File(os.path.join('training_results', 'baseline', 'mcp_history.h5'),
               'w') as hist_file:
  hist_file.create_dataset('loss', data=loss,
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('val_loss', data=val_loss,
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('epoch', data=epoch, compression='gzip',
                           compression_opts=9)

  hist_file.close()

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

exit()
