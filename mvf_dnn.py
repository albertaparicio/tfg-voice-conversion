"""
Created by albert aparicio on 18/10/16

This script is based off of the 'mnist_irnn' example from Keras
(https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py)

This is a script for initializing and training a fully-connected DNN for MVF mapping
"""

# TODO General code cleanup

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from sys import exit
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import RMSprop
# from keras.optimizers import SGD
from matplotlib import pyplot as plt

from error_metrics import RMSE

# from keras.datasets import mnist
# from keras.initializations import normal, identity
# from keras.layers import Activation
# from keras.layers import SimpleRNN
# from keras.utils import np_utils

# Switch to decide if datatable must be build or can be retrieved
build_datatable = False

print('Starting...')

# TODO Load proper data for a DNN training
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
    # Save and compress with .gz to save space (aprox 4x smaller file)
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
    # Save and compress with .gz to save space (aprox 4x smaller file)
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
batch_size = 300
# nb_classes = 10
nb_epochs = 50
# nb_epochs = 1000

# learning_rate = 1e-5
learning_rate = 0.001

# Split into train and validation (17500 train, 2500 validation)
# Randomize frames
np.random.shuffle(train_data)

# Split in training/test, picking only MVF and U/V data
src_train_frames = train_data[0:17500, 41:43]  # Source data
trg_train_frames = train_data[0:17500, 84:86]  # Target data

np.savetxt('src_train_frames.csv', src_train_frames, delimiter=',')
np.savetxt('trg_train_frames.csv', trg_train_frames, delimiter=',')

src_valid_frames = train_data[17500:train_data.shape[0], 41:43]  # Source data
trg_valid_frames = train_data[17500:train_data.shape[0], 84:86]  # Target data

# Remove means and normalize
src_train_mean = np.mean(src_train_frames[:, 0], axis=0)
src_train_std = np.std(src_train_frames[:, 0], axis=0)
trg_train_mean = np.mean(trg_train_frames[:, 0], axis=0)
trg_train_std = np.std(trg_train_frames[:, 0], axis=0)

src_train_frames[:, 0] = (src_train_frames[:, 0] - src_train_mean) / src_train_std
trg_train_frames[:, 0] = (trg_train_frames[:, 0] - trg_train_mean) / trg_train_std

# src_valid_frames[:, 0] = src_valid_frames[:, 0] - np.mean(src_valid_frames[:, 0], axis=0)
src_valid_frames[:, 0] = (src_valid_frames[:, 0] - src_train_mean) / src_train_std
# trg_valid_frames[:, 0] = trg_valid_frames[:, 0] - np.mean(trg_valid_frames[:, 0], axis=0)
trg_valid_frames[:, 0] = (trg_valid_frames[:, 0] - trg_train_mean) / trg_train_std

np.savetxt('src_train_frames_nomean.csv', src_train_frames, delimiter=',')
np.savetxt('trg_train_frames_nomean.csv', trg_train_frames, delimiter=',')
# exit()
# TODO Define a fully-connected DNN
print('Evaluate DNN...')
model = Sequential()

model.add(Dense(100, input_dim=2))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(2, activation='linear'))

# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, clipnorm=10)
rmsprop = RMSprop(lr=learning_rate)

# model.compile(loss='mse', optimizer=sgd)
model.compile(loss='mae', optimizer=rmsprop)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min')

# RMSE of training data
prediction = model.predict(src_train_frames)

prediction[:, 0] = (prediction[:, 0] * trg_train_std) + trg_train_mean
# src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

# Mask prediction data
vf_predictions = prediction[:, 0]
uv_predictions = np.round(prediction[:, 1])
vf_predictions[np.where(uv_predictions == 0)] = 1000

# Mask ground truth data
vf_gtruth = (trg_train_frames[:, 0] * trg_train_std) + trg_train_mean
uv_gtruth = np.round(trg_train_frames[:, 1])
vf_gtruth[np.where(uv_gtruth == 0)] = 1000

rmse_training_pre = RMSE(
    (trg_train_frames[:, 0] * trg_train_std) + trg_train_mean,
    prediction[:, 0],
    mask=trg_train_frames[:, 1]
)

print(rmse_training_pre)

history = model.fit(src_train_frames, trg_train_frames, batch_size=batch_size, nb_epoch=nb_epochs,
                    verbose=1, validation_data=(src_valid_frames, trg_valid_frames), callbacks=[reduce_lr])

# RMSE of training data
prediction = model.predict(src_train_frames)

prediction[:, 0] = (prediction[:, 0] * trg_train_std) + trg_train_mean
# src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

# Mask prediction data
vf_predictions = prediction[:, 0]
uv_predictions = np.round(prediction[:, 1])
vf_predictions[np.where(uv_predictions == 0)] = 1000

# Mask ground truth data
vf_gtruth = (trg_train_frames[:, 0] * trg_train_std) + trg_train_mean
uv_gtruth = np.round(trg_train_frames[:, 1])
vf_gtruth[np.where(uv_gtruth == 0)] = 1000

rmse_training_after = RMSE(
    (trg_train_frames[:, 0] * trg_train_std) + trg_train_mean,
    prediction[:, 0],
    mask=trg_train_frames[:, 1]
)

print(rmse_training_after)

model.save('mvf_dnn_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# plt.plot(history.epoch, history.history['loss'], history.epoch, history.history['val_loss'], '--', linewidth=2)
# plt.legend(['Training loss', 'Validation loss'])
# plt.grid(b='on')
# plt.savefig('losses.png', bbox_inches='tight')
# # plt.show()

np.savetxt('epoch.csv', history.epoch, delimiter=',')
np.savetxt('loss.csv', history.history['loss'], delimiter=',')
np.savetxt('val_loss.csv', history.history['val_loss'], delimiter=',')

# scores = model.evaluate(test_data[:, 41:43], test_data[:, 84:86], verbose=0)
#
# print(scores)

# # RMSE of test data
# test_mean = np.mean(test_data[:, 41], axis=0)
# test_std = np.std(test_data[:, 41], axis=0)
#
# test_data[:, 41] = (test_data[:, 41] - test_mean) / test_std
# prediction = model.predict(test_data[:, 41:43])
#
# prediction[:, 0] = (prediction[:, 0] * test_std) + test_mean
# test_data[:, 41] = (test_data[:, 41] * test_std) + test_mean
#
# rmse_res = RMSE(test_data[:, 41:43], prediction)

# RMSE of training data
prediction = model.predict(src_train_frames)

prediction[:, 0] = (prediction[:, 0] * src_train_std) + src_train_mean
src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

rmse_res = RMSE(src_train_frames, prediction)
# np.savetxt('rmse.csv', rmse_res, delimiter=',')

# print(rmse_res[0])

# # Histogram of predicted training data and training data itself
# plt.hist(prediction[:, 0], bins=100)
# plt.title('Prediction frames')
# plt.show()
#
# # Histogram of training samples
# plt.figure()
# plt.hist(src_train_frames[:, 0], bins=100)
# plt.title('Training frames')
# plt.show()

# TODO Print DNN scores
# print('IRNN test score:', scores[0])
# print('IRNN test accuracy:', scores[1])
exit()
