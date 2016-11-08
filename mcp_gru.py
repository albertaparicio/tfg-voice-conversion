# Created by albert aparicio on 06/11/16
# coding: utf-8

# This script defines a GRU-RNN to map the cepstral components of the signal

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.layers import GRU, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop

import utils

# Switch to decide if datatable must be build or can be loaded from a file
build_datatable = False

print('Starting...')

if build_datatable:
    import construct_table as ct

    # Build datatable of the training data
    # (data is already encoded with Ahocoder)
    print('Saving training datatable...', end='')
    train_data = ct.construct_datatable(
        'data/training/basenames.list',
        'data/training/vocoded/SF1/',
        'data/training/vocoded/TF1/',
        'data/training/frames/'
    )
    # Save and compress with .gz to save space (approx 4x smaller file)
    np.savetxt('data/train_datatable.csv.gz', train_data, delimiter=',')
    print('done')

    # Build datatable of the test data (data is already encoded with Ahocoder)
    print('Saving test datatable...', end='')
    test_data = ct.construct_datatable(
        'data/test/basenames.list',
        'data/test/vocoded/SF1/',
        'data/test/vocoded/TF1/',
        'data/test/frames/'
    )
    # Save and compress with .gz to save space (approx 4x smaller file)
    np.savetxt('data/test_datatable.csv.gz', test_data, delimiter=',')
    print('done')

else:
    # Retrieve datatable from .csv.gz file
    print('Loading training datatable...', end='')
    train_data = np.loadtxt('data/train_datatable.csv.gz', delimiter=',')
    print('done')
    print('Loading test datatable...', end='')
    test_data = np.loadtxt('data/test_datatable.csv.gz', delimiter=',')
    print('done')

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
lahead = 1  # number of elements ahead that are used to make the prediction

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
# trg_test_data = utils.reshape_lstm(trg_test_data, tsteps, data_dim)

# Save training statistics
# TODO migrate to h5py
# np.savetxt(
#     'mcp_train_stats.csv',
#     np.array(
#         [[src_train_mean, src_train_std],
#          [trg_train_mean, trg_train_std]]
#     ),
#     delimiter=','
# )

# exit()

################
# Define Model #
################
# Define an GRU-based RNN
print('Creating Model')
model = Sequential()

model.add(GRU(100,
              batch_input_shape=(batch_size, tsteps, data_dim),
              return_sequences=True,
              stateful=True))
model.add(TimeDistributed(Dense(data_dim)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)

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
model.save_weights('mcp_weights.h5')

with open('mcp_model.json', 'w') as model_json:
    model_json.write(model.to_json())

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
