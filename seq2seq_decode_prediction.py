# Created by Albert Aparicio on 7/12/16
# coding: utf-8

# This script takes a trained model and predicts the test data

# TODO subsitute print calls for logging.info calls when applicable
# https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import subprocess

import h5py
import numpy as np
import tfglib.seq2seq_datatable as s2s
import tfglib.seq2seq_normalize as s2s_norm
from ahoproc_tools import error_metrics
from keras.layers import Embedding
from keras.layers import Input, TimeDistributed, Dense, merge, Dropout
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
from tfglib.pretrain_data_params import prepare_pretrain_slice
from tfglib.pretrain_data_params import pretrain_load_data_parameters
from tfglib.pretrain_data_params import pretrain_save_data_parameters
from tfglib.utils import reverse_encoder_output, reversed_output_shape

############
# Switches #
############
pretrain = True

# Decide if datatable/parameters must be built or can be loaded from a file
build_datatable = False

#############
# Load data #
#############
if pretrain:
    data_path = 'pretrain_data/test'

    if build_datatable:
        print('Saving pretraining parameters')
        (max_test_length,
         test_speakers_max,
         test_speakers_min,
         test_files_list
         ) = pretrain_save_data_parameters(data_path)

    else:
        print('Loading pretraining parameters')
        (max_test_length,
         test_speakers_max,
         test_speakers_min,
         test_files_list
         ) = pretrain_load_data_parameters(data_path)

    test_speakers = test_speakers_max.shape[0]

else:
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

    test_speakers = test_speakers_max.shape[0]
    print('done')

#############################
# Load model and parameters #
#############################
model_description = 'seq2seq_pretrain'

print('Loading parameters')
with h5py.File('training_results/' + model_description + '_training_params.h5',
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
batch_size = 1

prediction_epoch = 19

#################
# Define models #
#################
# Encoder model
main_input = Input(batch_shape=(batch_size, max_test_length, output_dim),
                   dtype='float32',
                   name='main_input')

src_spk_input = Input(
    batch_shape=(batch_size, max_test_length,),
    dtype='int32',
    name='src_spk_in'
)
trg_spk_input = Input(
    batch_shape=(batch_size, max_test_length,),
    dtype='int32',
    name='trg_spk_in'
)

embedded_spk_indexes = Embedding(
    input_dim=test_speakers,
    output_dim=5,
    input_length=max_test_length,
    name='spk_index_embedding'
)

merged_parameters = merge(
    [main_input,
     embedded_spk_indexes(src_spk_input),
     embedded_spk_indexes(trg_spk_input)
     ],
    mode='concat',
    name='inputs_merge'
)

encoder_PLSTM = PLSTM(
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='encoder_PLSTM'
)(merged_parameters)

encoder_output = Lambda(
    reverse_encoder_output,
    output_shape=reversed_output_shape,
    name='reversed_encoder'
)(encoder_PLSTM)

# Decoder model
# decoder_input = Input(batch_shape=(batch_size, max_test_length, emb_size),
decoder_input = Input(batch_shape=(batch_size, 1, emb_size),
                      dtype='float32',
                      name='decoder_input')
feedback_input = Input(batch_shape=(batch_size, 1, output_dim),
                       name='feedback_in')
dec_in = merge([decoder_input, feedback_input],
               mode='concat',
               name='decoder_merge'
               )

decoder_PLSTM = PLSTM(
    emb_size,
    return_sequences=True,
    consume_less='gpu',
    stateful=True,
    name='decoder_PLSTM'
)(dec_in)

dropout_layer = Dropout(0.5)(decoder_PLSTM)

params_output = PLSTM(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear',
    stateful=True,
    name='params_output'
)(dropout_layer)

flags_output = TimeDistributed(Dense(
    2,
    activation='sigmoid',
), name='flags_output')(dropout_layer)

######################
# Instantiate models #
######################
encoder_model = Model(input=[main_input, src_spk_input, trg_spk_input],
                      output=encoder_output)
decoder_model = Model(input=[decoder_input, feedback_input],
                      output=[params_output, flags_output])

# Load weights and compile models
encoder_model.load_weights('models/' + model_description + '_' + params_loss +
                           '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
                           str(prediction_epoch) + '_lr_' + str(learning_rate) +
                           '_weights.h5', by_name=True)
decoder_model.load_weights('models/' + model_description + '_' + params_loss +
                           '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
                           str(prediction_epoch) + '_lr_' + str(learning_rate) +
                           '_weights.h5', by_name=True)

# load_weights(encoder_model, 'models/' + model_description + '_' + params_loss+
#              '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
#              str(prediction_epoch) + '_lr_' + str(learning_rate) +
#              '_weights.h5')
#
# load_weights(decoder_model, 'models/' + model_description + '_' + params_loss+
#              '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
#              str(prediction_epoch) + '_lr_' + str(learning_rate) +
#              '_weights.h5', offset=2)

adam = Adam(clipnorm=5)

encoder_model.compile(optimizer=adam,
                      loss={'reversed_encoder': params_loss},
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
if pretrain:
    # Initialize slices generator
    pretrain_slice = prepare_pretrain_slice(
        test_files_list,
        data_path,
        max_test_length,
        train_speakers_max,
        train_speakers_min,
        shuffle_files=False
    )

    # Initialize batch
    main_input = np.empty((batch_size, max_test_length, 44))
    src_spk_in = np.empty((batch_size, max_test_length))
    trg_spk_in = np.empty((batch_size, max_test_length))

    for sequence in test_files_list:
        print('Processing ' + sequence)

        # Get sequence parameters (only input parameters 'cos we have test data)
        (
            main_input[0, :, :],
            src_spk_in[0, :],
            trg_spk_in[0, :],
            _,
            _,
            _,
            _
        ) = next(pretrain_slice)

        # Predict parameters
        # ==================

        # Encoder prediction
        encoder_prediction = encoder_model.predict_on_batch({
            'main_input': main_input,
            'src_spk_in': src_spk_in,
            'trg_spk_in': trg_spk_in
        })

        # Prepare data for decoder predictions
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
        progress_bar = Progbar(target=max_test_length)
        progress_bar.update(0)

        # TODO Fix EOS prediction
        # while loop_timesteps < max_test_length:
        while EOS < 0.5 and loop_timesteps < max_test_length:

            # Predict each frame separately
            # for index in range(encoder_prediction.shape[1]):
            [partial_prediction[:, :, 0:42],
             partial_prediction[:, :, 42:44]
             ] = decoder_model.predict_on_batch(
                {'decoder_input': encoder_prediction[:, loop_timesteps, :].
                    reshape(1, -1, emb_size),
                 'feedback_in': feedback_data}
            )

            # Unscale partial prediction
            partial_prediction[
                :, :, 0:42
            ] = partial_prediction[:, :, 0:42].reshape(-1, 42) * (
                train_speakers_max[int(src_spk_in[0, loop_timesteps]), :] -
                train_speakers_min[int(src_spk_in[0, loop_timesteps]), :]
            ) + train_speakers_min[int(src_spk_in[0, loop_timesteps]), :]

            # Round U/V flag
            partial_prediction[:, :, 42] = np.round(
                partial_prediction[:, :, 42])

            # Apply u/v flags to lf0 and mvf
            if partial_prediction[:, :, 42] == 0:
                partial_prediction[:, :, 40] = -1e+10  # lf0
                partial_prediction[:, :, 41] = 1000  # mvf

            decoder_prediction = np.concatenate(
                (decoder_prediction, partial_prediction), axis=1)

            feedback_data = partial_prediction

            EOS = decoder_prediction[:, loop_timesteps, 43]
            loop_timesteps += 1

            progress_bar.update(loop_timesteps)

        # There is no need to un-zero-pad, since the while loop stops
        # when an EOS flag is found

        # Reshape prediction into 2D matrix
        decoder_prediction = decoder_prediction.reshape(-1, output_dim)
        raw_uv_flags = raw_uv_flags.reshape(-1, 1)

        # Save parameters to separate files #
        # Create destination directory before saving data
        predicted_path = 'predicted_' + sequence

        bashCommand = ('mkdir -p ' + predicted_path[:-11])
        process = subprocess.Popen(
            bashCommand.split(),
            stdout=subprocess.PIPE
        )
        output, error = process.communicate()

        np.savetxt(
            predicted_path + '.vf.dat',
            decoder_prediction[:, 41]
        )
        np.savetxt(
            predicted_path + '.lf0.dat',
            decoder_prediction[:, 40]
        )
        np.savetxt(
            predicted_path + '.mcp.dat',
            decoder_prediction[:, 0:40],
            delimiter='\t'
        )
        np.savetxt(
            predicted_path + '.uv.dat',
            raw_uv_flags
        )

        # Display MCD
        print('MCD = ' +
              str(error_metrics.MCD(
                  main_input[0, :, 0:40], decoder_prediction[:, 0:40]
              )))

else:
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
    assert len(basenames) == src_test_datatable.shape[0] / np.square(
        len(speakers))

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
                # TODO Fix EOS prediction
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
                        decoder_prediction[:, loop_timesteps, 0:42].reshape(-1,
                                                                            42),
                        train_speakers_max,
                        train_speakers_min
                    )

                    ###################
                    # Round u/v flags #
                    ###################
                    raw_uv_flags = np.append(
                        raw_uv_flags,
                        [decoder_prediction[:, loop_timesteps, 42]],
                        axis=0
                    )

                    decoder_prediction[:, loop_timesteps, 42] = np.round(
                        decoder_prediction[:, loop_timesteps, 42])

                    # Apply u/v flags to lf0 and mvf
                    # for index, entry in enumerate(prediction[:, 42]):
                    #     if entry == 0:
                    if decoder_prediction[:, loop_timesteps, 42] == 0:
                        decoder_prediction[:, loop_timesteps, 40] = -1e10  # lf0
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
                print('MCD = ' + str(error_metrics.MCD(
                    trg_test_datatable[
                        i + (src_spk_ind + trg_spk_ind) * len(basenames),
                        0:int(sum(trg_test_masks[
                              i + (src_spk_ind + trg_spk_ind) *
                                  len(basenames), :]
                                  )),
                        0:40
                    ],
                    decoder_prediction[
                        0:int(sum(trg_test_masks[
                              i + (src_spk_ind + trg_spk_ind) *
                                  len(basenames), :]
                                  )),
                        0:40
                    ]
                )))

            trg_spk_ind += 1
        src_spk_ind += 1
