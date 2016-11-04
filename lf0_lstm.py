# Created by albert aparicio on 31/10/16
# coding: utf-8

# This script initializes and trains an LSTM-based RNN for log(f0) mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.layers import LSTM, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

import utils
from error_metrics import RMSE

# Switch to decide if datatable must be build or can be loaded from a file
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
#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 1
tsteps = 50
data_dim = 2

# Other constants
epochs = 50
# epochs = 25
lahead = 1  # number of elements ahead that are used to make the prediction

################
# Prepare data #
################
# Take lfo and U/V flag columns
src_train_frames = np.column_stack((train_data[0:17500, 40], train_data[0:17500, 42]))  # Source data
trg_train_frames = np.column_stack((train_data[0:17500, 83], train_data[0:17500, 85]))  # Target data

src_valid_frames = np.column_stack(
    (train_data[17500:train_data.shape[0], 40], train_data[17500:train_data.shape[0], 42]))  # Source data
trg_valid_frames = np.column_stack(
    (train_data[17500:train_data.shape[0], 83], train_data[17500:train_data.shape[0], 85]))  # Target data

src_test_frames = np.column_stack((test_data[:, 40], test_data[:, 42]))
trg_test_frames = np.column_stack((test_data[:, 83], test_data[:, 85]))

# Remove means and normalize
src_train_mean = np.mean(src_train_frames[:, 0], axis=0)
src_train_std = np.std(src_train_frames[:, 0], axis=0)

src_train_frames[:, 0] = (src_train_frames[:, 0] - src_train_mean) / src_train_std
src_valid_frames[:, 0] = (src_valid_frames[:, 0] - src_train_mean) / src_train_std
src_test_frames[:, 0] = (src_test_frames[:, 0] - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_frames[:, 0], axis=0)
trg_train_std = np.std(trg_train_frames[:, 0], axis=0)

trg_train_frames[:, 0] = (trg_train_frames[:, 0] - trg_train_mean) / trg_train_std
trg_valid_frames[:, 0] = (trg_valid_frames[:, 0] - trg_train_mean) / trg_train_std
# trg_test_frames[:, 0] = (trg_test_frames[:, 0] - trg_train_mean) / trg_train_std

# Zero-pad and reshape data
src_train_frames = utils.reshape_lstm(src_train_frames, tsteps, data_dim)
src_valid_frames = utils.reshape_lstm(src_valid_frames, tsteps, data_dim)
src_test_frames = utils.reshape_lstm(src_test_frames, tsteps, data_dim)
trg_train_frames = utils.reshape_lstm(trg_train_frames, tsteps, data_dim)
trg_valid_frames = utils.reshape_lstm(trg_valid_frames, tsteps, data_dim)
trg_test_frames = utils.reshape_lstm(trg_test_frames, tsteps, data_dim)

# Save training statistics
np.savetxt(
    'lf0_train_stats.csv',
    np.array([[src_train_mean, src_train_std],[trg_train_mean, trg_train_std]]),
    delimiter=','
)

# exit()

################
# Define Model #
################
# Define an LSTM-based RNN
print('Creating Model')
model = Sequential()

model.add(LSTM(100,
               batch_input_shape=(batch_size, tsteps, data_dim),
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(2)))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(src_train_frames,
              trg_train_frames,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False,
              validation_data=(src_valid_frames, trg_valid_frames))
    model.reset_states()

print('Saving model')
# model.save('lf0_lstm_model.h5')
model.save_weights('lf0_weights.h5')

with open('lf0_model.json', 'w') as model_json:
    model_json.write(model.to_json())

print('Predicting')
prediction_test = model.predict(src_test_frames, batch_size=batch_size)

# De-normalize predicted output
prediction_test[:, 0] = (prediction_test[:, 0] * trg_train_std) + trg_train_mean

# Compute RMSE of test data
rmse_test = RMSE(np.exp(trg_test_frames[:, 0]), np.exp(prediction_test[:, 0]), mask=trg_test_frames[:, 1])

# Print resulting RMSE
print('Test RMSE: ', rmse_test)
