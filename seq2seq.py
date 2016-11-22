# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from keras.layers import GRU, Dropout
from keras.layers.core import RepeatVector
from keras.models import Sequential
from keras.optimizers import Adamax

from tfglib import construct_table as ct
from tfglib import utils

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 100
data_dim = 44 + 10 + 10

# # Other constants
# epochs = 50
# # epochs = 25
# lahead = 1  # number of elements ahead that are used to make the prediction
learning_rate = 0.01

#############
# Load data #
#############

################
# Prepare data #
################

################
# Define Model #
################
print('Initializing model')
model = Sequential()

# Encoder Layer
model.add(GRU(100,
              input_dim=44+2*10,
              return_sequences=False
              ))
model.add(RepeatVector(max_seq_length))

# Decoder layer
model.add(GRU(100, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(44, return_sequences=True, activation='linear'))

adamax = Adamax(lr=learning_rate, clipnorm=10)
model.compile(loss='mse', optimizer=adamax)

###############
# Train model #
###############

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
