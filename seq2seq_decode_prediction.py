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
from keras.layers import BatchNormalization, GRU, Dropout
from keras.layers import Input, TimeDistributed, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import RepeatVector
from keras.models import Model
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
with h5py.File('training_results/seq2seq_feedback_training_params.h5',
               'r') as f:
    params_loss = f.attrs.get('params_loss').decode('utf-8')
    flags_loss = f.attrs.get('flags_loss').decode('utf-8')
    optimizer_name = f.attrs.get('optimizer').decode('utf-8')
    nb_epochs = f.attrs.get('epochs')
    learning_rate = f.attrs.get('learning_rate')
    train_speakers_max = f.attrs.get('train_speakers_max')
    train_speakers_min = f.attrs.get('train_speakers_min')

print('Re-initializing model')
output_dim = 44
data_dim = output_dim + 10 + 10
emb_size = 256

main_input = Input(shape=(max_test_length, data_dim),
                   dtype='float32',
                   name='main_input')

emb_a = TimeDistributed(Dense(emb_size))(main_input)
emb_bn = BatchNormalization()(emb_a)
emb_h = LeakyReLU()(emb_bn)

encoder_GRU = GRU(
    output_dim=256,
    return_sequences=False,
    consume_less='gpu'
)(emb_h)

repeat_layer = RepeatVector(max_test_length)(encoder_GRU)

decoder_GRU = GRU(256, return_sequences=True, consume_less='gpu')(repeat_layer)

dropout_layer = Dropout(0.5)(decoder_GRU)

parameters_GRU = GRU(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear',
    name='params_output'
)(dropout_layer)

flags_Dense = TimeDistributed(Dense(
    2,
    activation='sigmoid',
), name='flags_output')(dropout_layer)

seq2seq_model = Model(input=main_input, output=[parameters_GRU, flags_Dense])

adam = Adam(clipnorm=5)
seq2seq_model.compile(optimizer=adam,
                      loss={'params_output': params_loss,
                            'flags_output': flags_loss},
                      sample_weight_mode="temporal"
                      )

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
            # ######################################################
            # ######################################################
            # ######################################################
            # ######################################################
            # ######################################################
            # it_sequence = src_test_datatable[i, :, :]
            #
            # # Test
            # prediction = np.empty((1, max_test_length, output_dim))
            #
            # [prediction[:, :, 0:42],
            #  prediction[:, :, 42:44]] = seq2seq_model.predict(
            #     it_sequence.reshape(1, -1, it_sequence.shape[1]))
            #
            # # Unscale parameters
            # prediction[:, :, 0:42] = s2s_norm.unscale_prediction(
            #     src_test_datatable[i, :, :],
            #     src_test_masks[i, :],
            #     prediction[:, :, 0:42].reshape(-1, 42),
            #     train_speakers_max,
            #     train_speakers_min
            # )
            #
            # # Reshape prediction into 2D matrix
            # prediction = prediction.reshape(-1, output_dim)
            # ######################################################

            prediction = trg_test_datatable[i, :, :]
            # ######################################################
            # ######################################################
            # ######################################################
            # ######################################################
            # ######################################################

            ###################
            # Round u/v flags #
            ###################
            prediction[:, 42] = np.round(prediction[:, 42])

            # Apply u/v flags to lf0 and mvf
            for index, entry in enumerate(prediction[:, 42]):
                if entry == 0:
                    prediction[index, 40] = -1e+10  # lf0
                    prediction[index, 41] = 0  # mvf

            ##################################################
            # Un-zero-pad according to End-Of-Sequence flags #
            ##################################################
            # Round EOS flags
            prediction[:, 43] = np.round(prediction[:, 43])

            # Find EOS flag
            print(np.sum(prediction[:, 43]))

            eos_flag_index = int(np.nonzero(prediction[:, 43])[0])

            # Remove all frames after the EOS flag
            prediction = prediction[0:eos_flag_index + 1, :]

            # Check that the last EOS parameter is the flag
            assert prediction[-1, 43] == 1

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
