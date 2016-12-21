# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from time import time

import h5py
import numpy as np
import tfglib.seq2seq_datatable as s2s
from keras.layers import BatchNormalization, GRU, Dropout
from keras.layers import Input, TimeDistributed, Dense, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from tfglib.seq2seq_normalize import maxmin_scaling
from tfglib.utils import display_time

# Save training start time
start_time = time()

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 100
output_dim = 44
data_dim = output_dim + 10 + 10
emb_size = 256

# Other constants
nb_epochs = 50
# lahead = 1  # number of elements ahead that are used to make the prediction
learning_rate = 0.001
validation_fraction = 0.25

#############
# Load data #
#############
# Switch to decide if datatable must be built or can be loaded from a file
build_datatable = False

print('Preparing data\n' + '=' * 8 * 5)
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

################################################
# Split data into training and validation sets #
################################################
# ###################################
# # TODO ELIMINATE AFTER DEVELOPING #
# ###################################
# batch_size = 2
# nb_epochs = 2
#
# num = 10
# src_train_datatable = src_train_datatable[0:num]
# src_train_masks = src_train_masks[0:num]
# trg_train_datatable = trg_train_datatable[0:num]
# trg_train_masks = trg_train_masks[0:num]
# #################################################

src_train_data = src_train_datatable[0:int(np.floor(
    src_train_datatable.shape[0] * (1 - validation_fraction)))]
src_valid_data = src_train_datatable[int(np.floor(
    src_train_datatable.shape[0] * (1 - validation_fraction))):]

trg_train_data = trg_train_datatable[0:int(np.floor(
    trg_train_datatable.shape[0] * (1 - validation_fraction)))]
trg_train_masks_f = trg_train_masks[0:int(np.floor(
    trg_train_masks.shape[0] * (1 - validation_fraction)))]

trg_valid_data = trg_train_datatable[int(np.floor(
    trg_train_datatable.shape[0] * (1 - validation_fraction))):]
trg_valid_masks_f = trg_train_masks[int(np.floor(
    trg_train_masks.shape[0] * (1 - validation_fraction))):]

################
# Define Model #
################
print('Initializing model\n' + '=' * 8 * 5)
main_input = Input(shape=(max_train_length, data_dim),
                   dtype='float32',
                   name='main_input')

emb_a = TimeDistributed(Dense(emb_size))(main_input)
emb_bn = BatchNormalization()(emb_a)
emb_h = LeakyReLU()(emb_bn)

encoder_GRU = GRU(
    output_dim=256,
    # input_shape=(max_train_length, data_dim),
    return_sequences=False,
    consume_less='gpu',
)(emb_h)
enc_ReLU = LeakyReLU()(encoder_GRU)

repeat_layer = RepeatVector(max_train_length)(enc_ReLU)

# Feedback input
feedback_in = Input(shape=(max_train_length, output_dim), name='feedback_in')
dec_in = merge([repeat_layer, feedback_in], mode='concat')

decoder_GRU = GRU(256, return_sequences=True, consume_less='gpu')(dec_in)
dec_ReLU = LeakyReLU()(decoder_GRU)

dropout_layer = Dropout(0.5)(dec_ReLU)

parameters_GRU = GRU(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear'
)(dropout_layer)
params_ReLU = LeakyReLU(name='params_output')(parameters_GRU)

flags_Dense = TimeDistributed(Dense(
    2,
    activation='sigmoid',
), name='flags_output')(dropout_layer)

model = Model(input=[main_input, feedback_in],
              output=[params_ReLU, flags_Dense])

optimizer_name = 'adam'
adam = Adam(clipnorm=5)
params_loss = 'mse'
flags_loss = 'binary_crossentropy'

model.compile(optimizer=adam,
              loss={'params_output': params_loss,
                    'flags_output': flags_loss},
              sample_weight_mode="temporal"
              )

###############
# Train model #
###############
print('Training\n' + '=' * 8 * 5)

training_history = []
validation_history = []

for epoch in range(nb_epochs):
    print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

    nb_batches = int(src_train_data.shape[0] / batch_size)
    progress_bar = Progbar(target=nb_batches)

    epoch_train_partial_loss = []

    try:
        progress_bar.update(0)
    except OverflowError as err:
        raise Exception('nb_batches is 0. Please check the training data')

    for index in range(nb_batches):
        # Get batch of sequences and masks
        src_batch = src_train_data[
                    index * batch_size:(index + 1) * batch_size]
        trg_batch = trg_train_data[
                    index * batch_size:(index + 1) * batch_size]
        batch_masks = trg_train_masks_f[
                      index * batch_size:(index + 1) * batch_size]

        # Prepare feedback data
        feedback_data = np.roll(trg_batch, 1, axis=2)
        feedback_data[:, 0, :] = 0

        epoch_train_partial_loss.append(
            model.train_on_batch(
                {'main_input': src_batch,
                 'feedback_in': feedback_data},
                {'params_output': trg_batch[:, :, 0:42],
                 'flags_output': trg_batch[:, :, 42:44]},
                sample_weight={'params_output': batch_masks,
                               'flags_output': batch_masks}
            )
        )

        progress_bar.update(index + 1)

    # Prepare validation feedback data
    feedback_valid_data = np.roll(trg_valid_data, 1, axis=2)
    feedback_valid_data[:, 0, :] = 0

    epoch_val_loss = model.evaluate(
        {'main_input': src_valid_data,
         'feedback_in': feedback_valid_data},
        {'params_output': trg_valid_data[:, :, 0:42],
         'flags_output': trg_valid_data[:, :, 42:44]},
        batch_size=batch_size,
        sample_weight={'params_output': trg_valid_masks_f,
                       'flags_output': trg_valid_masks_f},
        verbose=0
    )

    epoch_train_loss = np.mean(np.array(epoch_train_partial_loss), axis=0)

    training_history.append(epoch_train_loss)
    validation_history.append(epoch_val_loss)

    # Generate epoch report
    print('loss: ' + str(training_history[-1]) +
          ' - val_loss: ' + str(validation_history[-1]) +
          '\n'  # + '-' * 24
          )
    print()

###############
# Saving data #
###############
print('Saving model\n' + '=' * 8 * 5)
model.save_weights(
    'models/seq2seq_feedback_' + params_loss + '_' + flags_loss + '_' + optimizer_name +
    '_epochs_' + str(nb_epochs) + '_lr_' + str(learning_rate) + '_weights.h5')

with open('models/seq2seq_feedback_' + params_loss + '_' + flags_loss + '_' +
          optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
          str(learning_rate) + '_model.json', 'w'
          ) as model_json:
    model_json.write(model.to_json())

print('Saving training parameters\n' + '=' * 8 * 5)
with h5py.File('training_results/seq2seq_feedback_training_params.h5', 'w') as f:
    f.attrs.create('params_loss', np.string_(params_loss))
    f.attrs.create('flags_loss', np.string_(flags_loss))
    f.attrs.create('optimizer', np.string_(optimizer_name))
    f.attrs.create('epochs', nb_epochs, dtype=int)
    f.attrs.create('learning_rate', learning_rate)
    f.attrs.create('train_speakers_max', train_speakers_max)
    f.attrs.create('train_speakers_min', train_speakers_min)
    f.attrs.create(
        'metrics_names',
        [np.string_(name) for name in model.metrics_names]
    )

print('Saving training results')
np.savetxt('training_results/seq2seq_feedback_' + params_loss + '_' + flags_loss + '_' +
           optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
           str(learning_rate) + '_epochs.csv',
           np.arange(nb_epochs), delimiter=',')
np.savetxt('training_results/seq2seq_feedback_' + params_loss + '_' + flags_loss + '_' +
           optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
           str(learning_rate) + '_loss.csv',
           training_history, delimiter=',')
np.savetxt('training_results/seq2seq_feedback_' + params_loss + '_' + flags_loss + '_' +
           optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
           str(learning_rate) + '_val_loss.csv',
           validation_history, delimiter=',')

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

end_time = time()
print('Elapsed time: ' + display_time(end_time - start_time))

exit()
