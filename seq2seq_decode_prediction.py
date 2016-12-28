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
from keras.layers import Input, TimeDistributed, Dense, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam


# TODO Move the finished function to tfglib
def load_weights(model, filepath, mode):
    import h5py
    from keras import backend as K
    w_file = h5py.File(filepath, mode='r')

    # Model.load_weights_from_hdf5_group(model, file)
    if hasattr(model, 'flattened_layers'):
        # Support for legacy Sequential/Merge behavior.
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    if 'nb_layers' in w_file.attrs:
        # Legacy format.
        nb_layers = w_file.attrs['nb_layers']
        if nb_layers != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(nb_layers) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

        for k in range(nb_layers):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in
                       range(g.attrs['nb_params'])]
            flattened_layers[k].set_weights(weights)
    else:
        # New file format.
        filtered_layers = []
        indexes = []
        for index, layer in enumerate(flattened_layers):
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
                indexes.append(index)

        if mode is 'encoder':
            offset = 0

        elif mode is 'decoder':
            offset = 6

        else:
            raise Exception("Unrecognized mode. Please choose 'encoder' or" +
                            " 'decoder' as a mode")

        indexes = [index + offset for index in indexes]
        flattened_layers = filtered_layers

        # layer_names = [n.decode('utf8') for n in w_file.attrs['layer_names']]
        all_names = [n.decode('utf8') for n in w_file.attrs['layer_names']]
        layer_names = [all_names[index] for index in indexes]
        filtered_layer_names = []
        for name in layer_names:
            g = w_file[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names

        if len(layer_names) != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(len(layer_names)) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = w_file[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = flattened_layers[k]
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            if layer.__class__.__name__ == 'Convolution1D':
                # This is for backwards compatibility with
                # the old Conv1D weights format.
                w = weight_values[0]
                shape = w.shape
                if shape[:2] != (layer.filter_length, 1) or \
                        shape[3] != layer.nb_filter:
                    # Legacy shape:
                    # (self.nb_filter, input_dim, self.filter_length, 1)
                    assert shape[0] == layer.nb_filter and shape[2:] == (
                        layer.filter_length, 1)
                    w = np.transpose(w, (2, 3, 1, 0))
                    weight_values[0] = w
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)


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
batch_size = 1

#################
# Define models #
#################
# Encoder model
encoder_input = Input(batch_shape=(batch_size, max_test_length, data_dim),
                      # shape=(max_test_length, data_dim),
                      dtype='float32',
                      name='encoder_input')

emb_a = TimeDistributed(Dense(emb_size), name='encoder_td_dense')(encoder_input)
emb_bn = BatchNormalization(name='enc_batch_norm')(emb_a)
emb_h = LeakyReLU()(emb_bn)

encoder_GRU = GRU(
    output_dim=256,
    # input_shape=(batch_size, max_test_length, data_dim),
    # batch_input_shape=(batch_size, max_test_length, data_dim),
    return_sequences=False,
    consume_less='gpu',
    stateful=True,
    name='encoder_GRU'
)(emb_h)
encoder_output = LeakyReLU(name='encoder_output')(encoder_GRU)

# Decoder model
decoder_input = Input(batch_shape=(batch_size, 1, emb_size), dtype='float32',
                      name='decoder_input')
feedback_input = Input(batch_shape=(batch_size, 1, output_dim),
                       name='feedback_in')
dec_in = merge([decoder_input, feedback_input], mode='concat')

decoder_GRU = GRU(256, return_sequences=True, consume_less='gpu',
                  stateful=True, name='decoder_GRU')(dec_in)
dec_ReLU = LeakyReLU()(decoder_GRU)

dropout_layer = Dropout(0.5)(dec_ReLU)

parameters_GRU = GRU(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear',
    stateful=True,
    name='parameters_GRU'
)(dropout_layer)
params_output = LeakyReLU(name='params_output')(parameters_GRU)

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
             '_' + flags_loss + '_' + optimizer_name + '_epochs_' +
             str(nb_epochs) + '_lr_' + str(learning_rate) +
             '_weights.h5', 'encoder')

load_weights(decoder_model, 'models/seq2seq_feedback_' + params_loss +
             '_' + flags_loss + '_' + optimizer_name + '_epochs_' +
             str(nb_epochs) + '_lr_' + str(learning_rate) +
             '_weights.h5', 'decoder')

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

            # Feedback data for first decoder iteration
            feedback_data = np.zeros((batch_size, 1, output_dim))

            # Loop parameters
            loop_timesteps = 0
            EOS = 0
            max_loop = 1.5 * max_test_length

            # Decoder predictions
            while EOS < 0.5 or loop_timesteps < max_loop:
                print(loop_timesteps)

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
                decoder_prediction[:, loop_timesteps, 42] = np.round(
                    decoder_prediction[:, loop_timesteps, 42])

                # Apply u/v flags to lf0 and mvf
                # for index, entry in enumerate(prediction[:, 42]):
                #     if entry == 0:
                if decoder_prediction[:, loop_timesteps, 42] == 0:
                    decoder_prediction[:, loop_timesteps, 40] = -1e+10  # lf0
                    decoder_prediction[:, loop_timesteps, 41] = 0  # mvf

                #############################################
                # Concatenate prediction with feedback data #
                #############################################
                feedback_data = decoder_prediction[:, loop_timesteps,
                                                   :].reshape(1, -1, output_dim)

                EOS = decoder_prediction[:, loop_timesteps, 43]
                loop_timesteps += 1

            # There is no need to un-zero-pad, since the while loop stops
            # when an EOS flag is found

            # Reshape prediction into 2D matrix
            decoder_prediction = decoder_prediction.reshape(-1, output_dim)

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
