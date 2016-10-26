"""
Created by albert aparicio on 18/10/16

This script is based off of the 'mnist_irnn' example from Keras
(https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py)

This is a test script for initializing and training a fully-connected DNN
"""

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
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
nb_epochs = 100
hidden_units = 10

learning_rate = 1e-6
clip_norm = 1.0

# Split into train and validation (17500 train, 2500 validation)
# Randomize frames
np.random.shuffle(train_data)

# Split in training/test, picking only MVF and U/V data
src_train_frames = train_data[0:17500, 41:43]  # Source data
trg_train_frames = train_data[0:17500, 84:86]  # Target data

src_valid_frames = train_data[17500:train_data.shape[0], 41:43]  # Source data
trg_valid_frames = train_data[17500:train_data.shape[0], 84:86]  # Target data

# TODO Define a fully-connected DNN
print('Evaluate DNN...')
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='relu'))
# model.add(Activation('softmax'))
# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mse', optimizer=rmsprop)

history = model.fit(src_train_frames, trg_train_frames, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(src_valid_frames, trg_valid_frames))
np.savetxt('loss.csv', history.history['loss'], delimiter=',')
np.savetxt('val_loss.csv', history.history['val_loss'], delimiter=',')
# scores = model.evaluate(X_test, Y_test, verbose=0)
scores = model.evaluate(test_data[:, 41:43], test_data[:, 84:86], verbose=0)

# TODO Print DNN scores
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
