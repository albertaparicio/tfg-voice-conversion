"""
Created by albert aparicio on 18/10/16

This script is based off of the 'mnist_irnn' example from Keras
(https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py)

This is a test script for initializing and training a fully-connected DNN
"""

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.initializations import normal, identity
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

# Switch to decide if datatable must be build or can be retrieved
build_datatable = False

# TODO Load proper data for a DNN training
if build_datatable:
    import construct_table as ct

    # Build datatable of the training data (data is already encoded with Ahocoder)
    train_data = ct.construct_datatable(
        'data/training/basenames.list',
        'data/training/vocoded/SF1/',
        'data/training/vocoded/TF1/',
        'data/training/frames/'
    )
    # Save and compress with .gz to save space (aprox 4x smaller file)
    np.savetxt('data/train_datatable.csv.gz', train_data, delimiter=',')

    # Build datatable of the test data (data is already encoded with Ahocoder)
    test_data = ct.construct_datatable(
        'data/test/basenames.list',
        'data/test/vocoded/SF1/',
        'data/test/vocoded/TF1/',
        'data/test/frames/'
    )
    # Save and compress with .gz to save space (aprox 4x smaller file)
    np.savetxt('data/test_datatable.csv.gz', test_data, delimiter=',')

else:
    # Retrieve datatable from .csv.gz file
    train_data = np.loadtxt('data/train_datatable.csv.gz', delimiter=',')
    test_data = np.loadtxt('data/test_datatable.csv.gz', delimiter=',')

# TODO adjust sizes and other constants
batch_size = 32
nb_classes = 10
nb_epochs = 200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# Split into train and validation (17500 train, 2500 validation)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# TODO Define a fully-connected DNN
print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)

# TODO Print DNN scores
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
