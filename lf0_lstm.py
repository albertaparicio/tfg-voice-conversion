# Created by albert aparicio on 31/10/16
# coding: utf-8

# This script initializes and trains an LSTM-based RNN for log(f0) mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from keras.layers import LSTM, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop
from tfglib import construct_table as ct
from tfglib import utils

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 2

# Other constants
epochs = 50
lahead = 1  # number of elements ahead that are used to make the prediction

#############
# Load data #
#############
# Switch to decide if datatable must be build or can be loaded from a file
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
# Number of training samples
nb_samples = 14500
# Take lfo and U/V flag columns
src_train_data = np.column_stack(
    (train_data[0:nb_samples, 40],
     train_data[0:nb_samples, 42])
)  # Source data

trg_train_data = np.column_stack(
    (train_data[0:nb_samples, 83],
     train_data[0:nb_samples, 85])
)  # Target data

src_valid_data = np.column_stack(
    (train_data[nb_samples:train_data.shape[0], 40],
     train_data[nb_samples:train_data.shape[0], 42])
)  # Source data
trg_valid_data = np.column_stack(
    (train_data[nb_samples:train_data.shape[0], 83],
     train_data[nb_samples:train_data.shape[0], 85])
)  # Target data

src_test_data = np.column_stack((test_data[:, 40], test_data[:, 42]))
trg_test_data = np.column_stack((test_data[:, 83], test_data[:, 85]))

# Remove means and normalize
src_train_mean = np.mean(src_train_data[:, 0], axis=0)
src_train_std = np.std(src_train_data[:, 0], axis=0)

src_train_data[:, 0] = (src_train_data[:, 0] - src_train_mean) / src_train_std
src_valid_data[:, 0] = (src_valid_data[:, 0] - src_train_mean) / src_train_std
src_test_data[:, 0] = (src_test_data[:, 0] - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_data[:, 0], axis=0)
trg_train_std = np.std(trg_train_data[:, 0], axis=0)

trg_train_data[:, 0] = (trg_train_data[:, 0] - trg_train_mean) / trg_train_std
trg_valid_data[:, 0] = (trg_valid_data[:, 0] - trg_train_mean) / trg_train_std
# trg_test_data[:, 0] = (trg_test_data[:, 0] - trg_train_mean) / trg_train_std

# Zero-pad and reshape data
src_train_data = utils.reshape_lstm(src_train_data, tsteps, data_dim)
src_valid_data = utils.reshape_lstm(src_valid_data, tsteps, data_dim)
src_test_data = utils.reshape_lstm(src_test_data, tsteps, data_dim)
trg_train_data = utils.reshape_lstm(trg_train_data, tsteps, data_dim)
trg_valid_data = utils.reshape_lstm(trg_valid_data, tsteps, data_dim)
trg_test_data = utils.reshape_lstm(trg_test_data, tsteps, data_dim)

# Save training statistics
with h5py.File('models/lf0_train_stats.h5', 'w') as f:
    h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
    h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
    h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
    h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

    f.close()

# exit()

################
# Define Model #
################
# Define an LSTM-based RNN
# TODO Define 2-output net
print('Creating Model')
model = Sequential()

model.add(LSTM(100,
               batch_input_shape=(batch_size, tsteps, data_dim),
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(2)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)

###############
# Train model #
###############
print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(src_train_data,
              trg_train_data,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False,
              validation_data=(src_valid_data, trg_valid_data))
    model.reset_states()

print('Saving model')
model.save_weights('models/lf0_weights.h5')

with open('models/lf0_model.json', 'w') as model_json:
    model_json.write(model.to_json())

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
