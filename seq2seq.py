# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
import tfglib.seq2seq_datatable as s2s
from keras.layers import GRU, Dropout
from keras.layers.core import RepeatVector
from keras.models import Sequential
from keras.optimizers import Adamax
from tfglib.seq2seq_normalize import maxmin_scaling

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 300
data_dim = 44 + 10 + 10

# Other constants
epochs = 50
# lahead = 1  # number of elements ahead that are used to make the prediction
learning_rate = 0.0002
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
     max_train_length,
     train_speakers_max,
     train_speakers_min
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
     max_test_length,
     test_speakers_max,
     test_speakers_min
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
     max_train_length,
     train_speakers_max,
     train_speakers_min
     ) = s2s.seq2seq2_load_datatable(
        'data/seq2seq_train_datatable.h5'
    )
    print('done')

    print('Loading test datatable...', end='')
    (src_test_datatable,
     src_test_masks,
     trg_test_datatable,
     trg_test_masks,
     max_test_length,
     test_speakers_max,
     test_speakers_min
     ) = s2s.seq2seq2_load_datatable(
        'data/seq2seq_test_datatable.h5'
    )
    print('done')

##################
# Normalize data #
##################
# Iterate over sequence 'slices'
assert src_train_datatable.shape[0] == trg_train_datatable.shape[0]

for i in range(src_train_datatable.shape[0]):
    (
        src_train_datatable[i, :, 0:42],
        trg_train_datatable[i, :, 0:42]
    ) = maxmin_scaling(
        src_train_datatable[i, :, :],
        src_train_masks[i, :],
        trg_train_datatable[i, :, :],
        trg_train_masks[i, :],
        train_speakers_max,
        train_speakers_min
    )

assert src_test_datatable.shape[0] == trg_test_datatable.shape[0]

for i in range(src_test_datatable.shape[0]):
    (
        src_test_datatable[i, :, 0:42],
        trg_test_datatable[i, :, 0:42]
    ) = maxmin_scaling(
        src_test_datatable[i, :, :],
        src_test_masks[i, :],
        trg_test_datatable[i, :, :],
        trg_test_masks[i, :],
        test_speakers_max,
        test_speakers_min
    )

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

optimizer = 'adamax'
adamax = Adamax(lr=learning_rate, clipnorm=10)
loss = 'mse'
model.compile(loss=loss, optimizer=adamax, sample_weight_mode="temporal")

###############
# Train model #
###############
print('Training')
history = model.fit(src_train_datatable,
                    trg_train_datatable,
                    batch_size=batch_size,
                    verbose=1,
                    nb_epoch=epochs,
                    shuffle=False,
                    validation_split=validation_fraction,
                    sample_weight=trg_train_masks)

print('Saving model')
model.save_weights('models/seq2seq_weights.h5')

with open('models/seq2seq_model.json', 'w') as model_json:
    model_json.write(model.to_json())

print('Saving training results')
# np.savetxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
#            str(epochs) + '_lr_' + str(learning_rate) +
#            '_data.csv', [epochs, learning_rate, optimizer, loss], delimiter=',')
np.savetxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
           str(epochs) + '_lr_' + str(learning_rate) +
           '_epochs.csv', history.epoch, delimiter=',')
np.savetxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
           str(epochs) + '_lr_' + str(learning_rate) +
           '_loss.csv', history.history['loss'], delimiter=',')
np.savetxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
           str(epochs) + '_lr_' + str(learning_rate) +
           '_val_loss.csv', history.history['val_loss'], delimiter=',')

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )
