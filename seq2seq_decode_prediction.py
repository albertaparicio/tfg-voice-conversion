# Created by Albert Aparicio on 7/12/16
# coding: utf-8

# This script takes a trained model and predicts the test data

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import subprocess

import h5py
import numpy as np
import tfglib.seq2seq_datatable as s2s
import tfglib.seq2seq_normalize as s2s_norm
from keras.layers import GRU, Dropout
from keras.layers.core import RepeatVector
from keras.models import Sequential
from keras.optimizers import Adam

######################
# Load test database #
######################
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

#############################
# Load model and parameters #
#############################
with h5py.File('training_results/seq2seq_training_params.h5', 'r') as f:
    epochs = f.attrs.get('epochs')
    learning_rate = f.attrs.get('learning_rate')
    optimizer = f.attrs.get('optimizer')
    loss = f.attrs.get('loss')
    train_speakers_max = f.attrs.get('train_speakers_max')
    train_speakers_min = f.attrs.get('train_speakers_min')

print('Re-initializing model')
seq2seq_model = Sequential()

# Encoder Layer
seq2seq_model.add(GRU(100,
                      input_dim=44 + 10 + 10,
                      return_sequences=False,
                      consume_less='gpu'
                      ))
seq2seq_model.add(RepeatVector(max_test_length))

# Decoder layer
seq2seq_model.add(GRU(100, return_sequences=True, consume_less='gpu'))
seq2seq_model.add(Dropout(0.5))
seq2seq_model.add(GRU(
    44,
    return_sequences=True,
    consume_less='gpu',
    activation='linear'
))

seq2seq_model.load_weights('models/seq2seq_' + loss.decode('utf-8') + '_' +
                           optimizer.decode('utf-8') + '_epochs_' +
                           str(epochs) + '_lr_' + str(learning_rate) +
                           '_weights.h5')

adam = Adam(clipnorm=10)
seq2seq_model.compile(loss=loss.decode('utf-8'), optimizer=adam,
                      sample_weight_mode="temporal")

##################
# Load basenames #
##################
basenames_file = open('data/test/seq2seq_basenames.list', 'r')
basenames_lines = basenames_file.readlines()
# Strip '\n' characters
basenames = [line.split('\n')[0] for line in basenames_lines]

# Load speakers
speakers_file = open('data/test/speakers.list', 'r')
speakers_lines = speakers_file.readlines()
# Strip '\n' characters
speakers = [line.split('\n')[0] for line in speakers_lines]

#######################
# Loop over sequences #
#######################
assert len(basenames) == src_test_datatable.shape[0] / np.square(len(speakers))

for src_spk in speakers:
    for trg_spk in speakers:
        # for i in range(src_test_datatable.shape[0]):
        for i in range(len(basenames)):
            ##################
            # Normalize data #
            ##################
            src_test_datatable[i, :, 0:42] = s2s_norm.maxmin_scaling(
                src_test_datatable[i, :, :],
                src_test_masks[i, :],
                trg_test_datatable[i, :, :],
                trg_test_masks[i, :],
                train_speakers_max,
                train_speakers_min
            )[0]

            # #################
            # # Mask sequence #
            # #################
            # masked_sequence = s2s_norm.mask_data(
            #     src_test_datatable[i, :, :],
            #     src_test_masks[i, :]
            # )
            #
            # #######################
            # # Get only valid data #
            # #######################
            # valid_sequence = masked_sequence[~masked_sequence.mask].reshape(
            #     (1,
            #      -1,
            #      masked_sequence.shape[1])
            # )

            ######################
            # Predict parameters #
            ######################
            it_sequence = src_test_datatable[i, :, :]
            prediction = seq2seq_model.predict(
                it_sequence.reshape(1, -1, it_sequence.shape[1]))

            ######################
            # Unscale parameters #
            ######################
            prediction[:, :, 0:42] = s2s_norm.unscale_prediction(
                src_test_datatable[i, :, :],
                src_test_masks[i, :],
                prediction[:, :, 0:42].reshape(-1, 42),
                train_speakers_max,
                train_speakers_min
            )

            #####################################
            # Reshape prediction into 2D matrix #
            #####################################
            prediction = prediction.reshape(-1, 44)

            ###################
            # Round u/v flags #
            ###################
            prediction[:, 42] = np.round(prediction[:, 42])

            ##################################
            # Apply u/v flags to lf0 and mvf #
            ##################################
            for index, entry in enumerate(prediction[:, 42]):
                if entry == 0:
                    prediction[index, 40] = -1e+10  # lf0
                    prediction[index, 41] = 0  # mvf

            #####################################
            # Save parameters to separate files #
            #####################################
            # Create destination directory before saving data
            bashCommand = ('mkdir -p data/test/s2s_predicted/' +
                           src_spk + '-' + trg_spk + '/')
            process = subprocess.Popen(
                bashCommand.split(),
                stdout=subprocess.PIPE
            )
            output, error = process.communicate()

            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.vf.dat',
                prediction[:, 41]
            )
            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.lf0.dat',
                prediction[:, 40]
            )
            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.mcp.dat',
                prediction[:, 0:40],
                delimiter='\t'
            )
