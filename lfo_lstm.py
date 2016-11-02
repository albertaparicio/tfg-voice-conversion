# Created by albert aparicio on 31/10/16
# coding: utf-8

# This script initializes and trains an LSTM-based RNN for log(f0) mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Switch to decide if datatable must be build or can be retrieved
build_datatable = False

print('Starting...')

if build_datatable:
    import construct_table as ct

    # Build datatable of the training data (data is already encoded with Ahocoder)
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

# TODO adjust sizes and other constants
'''Sizes and constants'''
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 2

# Other constants
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1

# TODO Prepare data for stateful LSTM RNN
# Take lfo and U/V flag columns
src_train_frames = np.column_stack((train_data[0:17500, 40], train_data[0:17500, 42]))  # Source data
trg_train_frames = np.column_stack((train_data[0:17500, 83], train_data[0:17500, 85]))  # Target data

src_valid_frames = np.column_stack(
    (train_data[17500:train_data.shape[0], 40], train_data[17500:train_data.shape[0], 42]))  # Source data
trg_valid_frames = np.column_stack(
    (train_data[17500:train_data.shape[0], 83], train_data[17500:train_data.shape[0], 85]))  # Target data

# Remove means and normalize
src_train_mean = np.mean(src_train_frames[:, 0], axis=0)
src_train_std = np.std(src_train_frames[:, 0], axis=0)
src_train_frames[:, 0] = (src_train_frames[:, 0] - src_train_mean) / src_train_std
src_valid_frames[:, 0] = (src_valid_frames[:, 0] - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_frames[:, 0], axis=0)
trg_train_std = np.std(trg_train_frames[:, 0], axis=0)
trg_train_frames[:, 0] = (trg_train_frames[:, 0] - trg_train_mean) / trg_train_std
trg_valid_frames[:, 0] = (trg_valid_frames[:, 0] - trg_train_mean) / trg_train_std

exit()

# TODO Define a fully-connected DNN
print('Creating Model')
model = Sequential()

model.add(LSTM(100,
               batch_input_shape=(batch_size, tsteps, data_dim),
               stateful=True))
model.add(LSTM(100,
               batch_input_shape=(batch_size, tsteps, data_dim),
               stateful=True))
model.add(Dense(2))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    # TODO Add validation data to 'fit'
    model.fit(src_train_frames,
              trg_train_frames,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Saving model')
model.save('lfo_lstm_model.h5')
