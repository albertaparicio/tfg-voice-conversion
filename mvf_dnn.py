# Created by albert aparicio on 18/10/16
# coding: utf-8

# This is a script for initializing and training a fully-connected DNN for MVF mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import RMSprop

from tfglib import construct_table as ct
from tfglib.utils import apply_context

#############
# Load data #
#############
#  Switch to decide if datatable must be build or can be loaded from a file
build_datatable = True

print('Starting...')

if build_datatable:
    # Build datatable of training and test data
    # (data is already encoded with Ahocoder)
    print('Saving training datatable...', end='')
    ct.save_datatable(
        'data/training/',
        'train_data',
        'data/train_datatable'
    )
    print('done')

    print('Saving test datatable...', end='')
    ct.save_datatable(
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

#######################
# Sizes and constants #
#######################
batch_size = 300
nb_epochs = 50
learning_rate = 0.001
context_size = 1

################
# Prepare data #
################
# Randomize frames
# np.random.shuffle(train_data)

# Split into train and validation (17500 train, 2500 validation)
src_train_frames = train_data[0:17500, 41:43]  # Source data
trg_train_frames = train_data[0:17500, 84:86]  # Target data

src_valid_frames = train_data[17500:train_data.shape[0], 41:43]  # Source data
trg_valid_frames = train_data[17500:train_data.shape[0], 84:86]  # Target data

# Normalize data
src_train_mean = np.mean(src_train_frames[:, 0], axis=0)
src_train_std = np.std(src_train_frames[:, 0], axis=0)
trg_train_mean = np.mean(trg_train_frames[:, 0], axis=0)
trg_train_std = np.std(trg_train_frames[:, 0], axis=0)

src_train_frames[:, 0] = (src_train_frames[:, 0] - src_train_mean) / src_train_std
src_valid_frames[:, 0] = (src_valid_frames[:, 0] - src_train_mean) / src_train_std

trg_train_frames[:, 0] = (trg_train_frames[:, 0] - trg_train_mean) / trg_train_std
trg_valid_frames[:, 0] = (trg_valid_frames[:, 0] - trg_train_mean) / trg_train_std

# Save training statistics
with h5py.File('models/mvf_train_stats.h5', 'w') as f:
    h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
    h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
    h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
    h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

    f.close()

# Apply context
src_train_frames_context = np.column_stack((
    apply_context(src_train_frames[:, 0], context_size), src_train_frames[:, 1]
))
src_valid_frames_context = np.column_stack((
    apply_context(src_valid_frames[:, 0], context_size), src_valid_frames[:, 1]
))

# exit()

################
# Define Model #
################
# Adjust DNN sizes to implement context
print('Evaluate DNN...')
model = Sequential()

# model.add(Dense(100, input_dim=2))
model.add(Dense(100, input_dim=(2 * context_size + 1) + 1))
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

###############
# Train model #
###############
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    mode='min')

history = model.fit(
    src_train_frames_context,
    trg_train_frames,
    batch_size=batch_size,
    nb_epoch=nb_epochs,
    verbose=1,
    validation_data=(src_valid_frames_context, trg_valid_frames),
    callbacks=[reduce_lr]
)

print('Saving model')
model.save_weights('models/mvf_weights.h5')

with open('models/mvf_model.json', 'w') as model_json:
    model_json.write(model.to_json())

print('========================' +
      '\n' +
      '======= FINISHED =======' +
      '\n' +
      '========================'
      )

exit()
