# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import tfglib.seq2seq_datatable as s2s
from keras.layers import GRU, Dropout
from keras.layers.core import RepeatVector
from keras.models import Sequential
from keras.optimizers import Adamax

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 100
data_dim = 44 + 10 + 10

# Other constants
epochs = 50
# lahead = 1  # number of elements ahead that are used to make the prediction
learning_rate = 0.01
validation_fraction = 0.25

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
    (src_train_datatable,
     src_train_masks,
     trg_train_datatable,
     trg_train_masks,
     max_train_length
     ) = s2s.seq2seq_save_datatable(
        'data/training/',
        'data/seq2seq_train_datatable'
    )
    print('done')

    print('Saving test datatable...', end='')
    (src_test_datatable,
     src_test_masks,
     trg_test_datatable,
     trg_test_masks,
     max_test_length
     ) = s2s.seq2seq_save_datatable(
        'data/test/',
        'data/seq2seq_test_datatable'
    )
    print('done')

else:
    # Retrieve datatables from .h5 files
    print('Loading training datatable...', end='')
    (src_train_datatable,
     src_train_masks,
     trg_train_datatable,
     trg_train_masks,
     max_train_length
     ) = s2s.seq2seq2_load_datatable(
        'data/seq2seq_train_datatable.h5'
    )
    print('done')

    print('Loading test datatable...', end='')
    (src_test_datatable,
     src_test_masks,
     trg_test_datatable,
     trg_test_masks,
     max_test_length
     ) = s2s.seq2seq2_load_datatable(
        'data/seq2seq_test_datatable.h5'
    )
    print('done')

################
# Define Model #
################
print('Initializing model')
model = Sequential()

# Encoder Layer
model.add(GRU(100,
              input_dim=data_dim,
              return_sequences=False
              ))
model.add(RepeatVector(max_train_length))

# Decoder layer
model.add(GRU(100, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(44, return_sequences=True, activation='linear'))

adamax = Adamax(lr=learning_rate, clipnorm=10)
model.compile(loss='mse', optimizer=adamax)

###############
# Train model #
###############
print('Training')
model.fit(src_train_datatable,
          trg_train_datatable,
          batch_size=batch_size,
          verbose=1,
          nb_epoch=epochs,
          shuffle=False,
          # validation_data=(src_valid_data, trg_valid_data),
          validation_split=validation_fraction,
          sample_weight=trg_train_masks)

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
