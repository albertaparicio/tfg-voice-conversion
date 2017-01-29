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
from ahoproc_tools import error_metrics
from keras.layers import Input, TimeDistributed, Dense, merge, Dropout
# from keras.layers import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
from tfglib.utils import s2s_load_weights as load_weights

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
print('Loading parameters')
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
emb_size = 1024
batch_size = 1

prediction_epoch = 24

#################
# Define models #
#################
# Encoder model
encoder_input = Input(batch_shape=(batch_size, max_test_length, data_dim),
                      # shape=(max_test_length, data_dim),
                      dtype='float32',
                      name='encoder_input')

# emb_a = TimeDistributed(Dense(256), name='encoder_td_dense')(encoder_input)
# emb_bn = BatchNormalization(name='enc_batch_norm')(emb_a)
# emb_h = LeakyReLU()(emb_bn)

encoder_output = PLSTM(
    output_dim=emb_size,
    # input_shape=(batch_size, max_test_length, data_dim),
    # batch_input_shape=(batch_size, max_test_length, data_dim),
    return_sequences=False,
    consume_less='gpu',
    stateful=True,
    name='encoder_output'
)(encoder_input)
# encoder_output = LeakyReLU(name='encoder_output')(encoder_PLSTM)

# Decoder model
decoder_input = Input(batch_shape=(batch_size, 1, emb_size), dtype='float32',
                      name='decoder_input')
feedback_input = Input(batch_shape=(batch_size, 1, output_dim),
                       name='feedback_in')
dec_in = merge([decoder_input, feedback_input], mode='concat')

decoder_PLSTM = PLSTM(
    256,
    return_sequences=True,
    consume_less='gpu',
    stateful=True,
    name='decoder_PLSTM'
)(dec_in)
# dec_ReLU = LeakyReLU()(decoder_PLSTM)

dropout_layer = Dropout(0.5)(decoder_PLSTM)

params_output = PLSTM(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear',
    stateful=True,
    name='params_output'
    # name='parameters_PLSTM'
)(dropout_layer)
# params_output = LeakyReLU(name='params_output')(parameters_PLSTM)

flags_output = TimeDistributed(Dense(
    2,
    activation='sigmoid',
), name='flags_output')(dropout_layer)

######################
# Instantiate models #
######################
encoder_model = Model(input=encoder_input,
                      output=encoder_output)
decoder_model = Model(input=[decoder_input, feedback_input],
                      output=[params_output, flags_output])

# Load weights and compile models
load_weights(encoder_model, 'models/seq2seq_feedback_' + params_loss +
             '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
             str(prediction_epoch) + '_lr_' + str(learning_rate) +
             '_weights.h5')

load_weights(decoder_model, 'models/seq2seq_feedback_' + params_loss +
             '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
             str(prediction_epoch) + '_lr_' + str(learning_rate) +
             '_weights.h5', offset=2)

adam = Adam(clipnorm=5)

encoder_model.compile(optimizer=adam,
                      loss={'encoder_output': params_loss},
                      sample_weight_mode="temporal"
                      )

decoder_model.compile(optimizer=adam,
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
print('Predicting sequences')
assert len(basenames) == src_test_datatable.shape[0] / np.square(len(speakers))

src_spk_ind = 0
trg_spk_ind = 0

for src_spk in speakers:
    for trg_spk in speakers:
        # for i in range(src_test_datatable.shape[0]):
        for i in range(len(basenames)):
            print(src_spk + '->' + trg_spk + ' ' + basenames[i])

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

            ######################
            # Predict parameters #
            ######################
            # Initialize encoder prediction data
            # ==================================
            it_sequence = src_test_datatable[i, :, :]
            src_batch = it_sequence.reshape(batch_size, -1,
                                            it_sequence.shape[1])

            # Encoder prediction
            # ==================
            encoder_prediction = encoder_model.predict_on_batch(src_batch)

            # Prepare data for decoder predictions
            # ====================================
            decoder_prediction = np.empty((batch_size, 0, output_dim))
            partial_prediction = np.empty((batch_size, 1, output_dim))
            raw_uv_flags = np.empty((0, 1))

            # Feedback data for first decoder iteration
            feedback_data = np.zeros((batch_size, 1, output_dim))

            # Loop parameters
            loop_timesteps = 0
            EOS = 0
            max_loop = 1.5 * max_test_length

            # Decoder predictions
            while loop_timesteps < max_test_length:
                # while EOS < 0.5 and loop_timesteps < max_loop:
                # print(loop_timesteps)

                [partial_prediction[:, :, 0:42],
                 partial_prediction[:, :, 42:44]
                 ] = decoder_model.predict_on_batch(
                    {'decoder_input': encoder_prediction.reshape(1, -1,
                                                                 emb_size),
                     'feedback_in': feedback_data}
                )

                decoder_prediction = np.concatenate(
                    (decoder_prediction, partial_prediction),
                    axis=1
                )

                # Unscale parameters
                decoder_prediction[
                    :, loop_timesteps, 0:42
                ] = s2s_norm.unscale_prediction(
                    src_test_datatable[i, :, :],
                    src_test_masks[i, :],
                    decoder_prediction[:, loop_timesteps, 0:42].reshape(-1, 42),
                    train_speakers_max,
                    train_speakers_min
                )

                ###################
                # Round u/v flags #
                ###################
                raw_uv_flags = np.append(
                    raw_uv_flags, [decoder_prediction[:, loop_timesteps, 42]],
                    axis=0
                )

                decoder_prediction[:, loop_timesteps, 42] = np.round(
                    decoder_prediction[:, loop_timesteps, 42])

                # Apply u/v flags to lf0 and mvf
                # for index, entry in enumerate(prediction[:, 42]):
                #     if entry == 0:
                if decoder_prediction[:, loop_timesteps, 42] == 0:
                    decoder_prediction[:, loop_timesteps, 40] = -1e+10  # lf0
                    decoder_prediction[:, loop_timesteps, 41] = 1000  # mvf

                #############################################
                # Concatenate prediction with feedback data #
                #############################################
                feedback_data = decoder_prediction[
                                :, loop_timesteps, :
                                ].reshape(1, -1, output_dim)

                EOS = decoder_prediction[:, loop_timesteps, 43]
                loop_timesteps += 1

            # There is no need to un-zero-pad, since the while loop stops
            # when an EOS flag is found

            # Reshape prediction into 2D matrix
            decoder_prediction = decoder_prediction.reshape(-1, output_dim)
            raw_uv_flags = raw_uv_flags.reshape(-1, 1)

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
                decoder_prediction[:, 41]
            )
            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.lf0.dat',
                decoder_prediction[:, 40]
            )
            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.mcp.dat',
                decoder_prediction[:, 0:40],
                delimiter='\t'
            )
            np.savetxt(
                'data/test/s2s_predicted/' + src_spk + '-' + trg_spk + '/' +
                basenames[i] + '.uv.dat',
                raw_uv_flags
            )

            # Display MCD
            print('MCD = ' +
                  str(error_metrics.MCD(
                      trg_test_datatable[
                        i + (src_spk_ind + trg_spk_ind) * len(basenames),
                        0:int(sum(trg_test_masks[
                                i + (src_spk_ind + trg_spk_ind) * len(
                                    basenames), :])),
                        0:40
                      ],
                      decoder_prediction[
                        0:int(sum(trg_test_masks[
                                i + (src_spk_ind + trg_spk_ind) * len(
                                    basenames), :])),
                        0:40
                      ]
                  ))
                  )

        trg_spk_ind += 1
    src_spk_ind += 1
