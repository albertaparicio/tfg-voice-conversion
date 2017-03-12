# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# TODO Document and explain steps
# TODO substitute print calls for logging.info calls when applicable
# https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from sys import version_info as py_version
from time import time

import h5py
import numpy as np
import tfglib.seq2seq_datatable as s2s
from ahoproc_tools import error_metrics
from keras.layers import Embedding
from keras.layers import Input, Dropout, Dense, merge, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from tfglib.pretrain_data_params import pretrain_load_data_parameters
from tfglib.pretrain_data_params import pretrain_save_data_parameters
from tfglib.pretrain_data_params import pretrain_train_generator
from tfglib.seq2seq_normalize import maxmin_scaling
from tfglib.utils import display_time

# Save training start time
start_time = time()

############
# Switches #
############
pretrain = True

# Decide if datatable/parameters must be built or can be loaded from a file
build_datatable = False

# Decide if the model must use weights from a previous training
load_weights = False

#######################
# Sizes and constants #
#######################
model_description = 'seq2seq_pretrain_double_model'

# Batch shape
batch_size = 40
output_dim = 44
data_dim = output_dim + 10 + 10
emb_size = 256

# Other constants
nb_epochs = 20
learning_rate = 0.001
validation_fraction = 0.25

if load_weights:
    start_epoch = 5
else:
    start_epoch = 0

#############
# Load data #
#############
if pretrain:
    data_path = 'pretrain_data/training_chop'

    if build_datatable:
        print('Saving pretraining parameters')
        (
            max_train_length,
            train_speakers_max,
            train_speakers_min,
            files_list
        ) = pretrain_save_data_parameters(data_path)

    else:
        print('Load pretraining parameters')
        (
            max_train_length,
            train_speakers_max,
            train_speakers_min,
            files_list
        ) = pretrain_load_data_parameters(data_path)

    train_speakers = train_speakers_max.shape[0]

    # #################################
    # # TODO COMMENT AFTER DEVELOPING #
    # #################################
    # batch_size = 2
    # nb_epochs = 1
    #
    # files_list = files_list[:5]
    #
    # model_description = 'DEV_' + model_description
    # #################################

else:
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

    train_speakers = train_speakers_max.shape[0]

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
    # #################################
    # # TODO COMMENT AFTER DEVELOPING #
    # #################################
    # batch_size = 2
    # nb_epochs = 2
    #
    # num = 10
    # src_train_datatable = src_train_datatable[0:num]
    # src_train_masks = src_train_masks[0:num]
    # trg_train_datatable = trg_train_datatable[0:num]
    # trg_train_masks = trg_train_masks[0:num]
    #
    # model_description = 'DEV_' + model_description
    # #################################################

    src_train_data = src_train_datatable[0:int(np.floor(
        src_train_datatable.shape[0] * (1 - validation_fraction)))]
    src_valid_data = src_train_datatable[int(np.floor(
        src_train_datatable.shape[0] * (1 - validation_fraction))):]

    trg_train_data = trg_train_datatable[0:int(np.floor(
        trg_train_datatable.shape[0] * (1 - validation_fraction)))]
    trg_valid_data = trg_train_datatable[int(np.floor(
        trg_train_datatable.shape[0] * (1 - validation_fraction))):]

    trg_train_masks_f = trg_train_masks[0:int(np.floor(
        trg_train_masks.shape[0] * (1 - validation_fraction)))]
    trg_valid_masks_f = trg_train_masks[int(np.floor(
        trg_train_masks.shape[0] * (1 - validation_fraction))):]

################
# Define Model #
################
print('Initializing model\n' + '=' * 8 * 5)

src_spk_input = Input(
    shape=(max_train_length,),
    dtype='int32',
    name='src_spk_in'
)
trg_spk_input = Input(
    shape=(max_train_length,),
    dtype='int32',
    name='trg_spk_in'
)

embedded_spk_indexes = Embedding(
    input_dim=train_speakers,
    output_dim=5,
    input_length=max_train_length,
    name='spk_index_embedding'
)

# Parameters model
parameters_input = Input(shape=(max_train_length, output_dim - 2),
                         dtype='float32',
                         name='main_input')

merged_parameters = merge(
    [parameters_input,
     embedded_spk_indexes(src_spk_input),
     embedded_spk_indexes(trg_spk_input)
     ],
    mode='concat',
    name='inputs_merge'
)

# Bidirectional encoder LSTM
params_encoder_LSTM_forwards = LSTM(
    go_backwards=False,
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='encoder_LSTM_forwards'
)(merged_parameters)

params_encoder_LSTM_backwards = LSTM(
    go_backwards=True,
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='encoder_LSTM_backwards'
)(merged_parameters)

params_encoder_LSTM_merged = merge(
    [params_encoder_LSTM_forwards, params_encoder_LSTM_backwards],
    mode='sum',
    name='encoder_bidirectional_merge'
)

# Feedback input
feedback_params_in = Input(shape=(max_train_length, output_dim - 2),
                           name='feedback_in')
dec_params_in = merge([params_encoder_LSTM_merged, feedback_params_in],
                      mode='concat')

decoder_params_LSTM = LSTM(
    emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='decoder_LSTM'
)(dec_params_in)

params_dropout_layer = Dropout(0.5)(decoder_params_LSTM)

parameters_LSTM = LSTM(
    output_dim - 2,
    return_sequences=True,
    consume_less='gpu',
    activation='linear',
    name='params_output'
)(params_dropout_layer)

# Flags model
flags_input = Input(shape=(max_train_length, 2),
                    dtype='float32',
                    name='main_input')

merged_flags = merge(
    [flags_input,
     embedded_spk_indexes(src_spk_input),
     embedded_spk_indexes(trg_spk_input)
     ],
    mode='concat',
    name='inputs_merge'
)

# Bidirectional encoder LSTM
flags_encoder_LSTM_forwards = LSTM(
    go_backwards=False,
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='encoder_LSTM_forwards'
)(merged_flags)

flags_encoder_LSTM_backwards = LSTM(
    go_backwards=True,
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='encoder_LSTM_backwards'
)(merged_flags)

flags_encoder_LSTM_merged = merge(
    [flags_encoder_LSTM_forwards, flags_encoder_LSTM_backwards],
    mode='sum',
    name='encoder_bidirectional_merge'
)

# Feedback input
feedback_flags_in = Input(shape=(max_train_length, 2),
                          name='feedback_in')
dec_flags_in = merge([flags_encoder_LSTM_merged, feedback_flags_in],
                     mode='concat')

decoder_flags_LSTM = LSTM(
    emb_size,
    return_sequences=True,
    consume_less='gpu',
    name='decoder_LSTM'
)(dec_flags_in)

dropout_flags_layer = Dropout(0.5)(decoder_flags_LSTM)

flags_Dense = TimeDistributed(Dense(
    2,
    activation='sigmoid',
), name='flags_output')(dropout_flags_layer)

params_model = Model(
    input=[parameters_input, src_spk_input, trg_spk_input, feedback_params_in],
    output=[parameters_LSTM])

flags_model = Model(
    input=[flags_input, src_spk_input, trg_spk_input, feedback_flags_in],
    output=[flags_Dense])

optimizer_name = 'adam'
adam = Adam(clipnorm=5)
params_loss = 'mse'
flags_loss = 'binary_crossentropy'

# if load_weights:
#     # Load weights from previous training
#     model.load_weights('models/' + model_description + '_' + params_loss +
#                        '_' + flags_loss + '_' + optimizer_name + '_epoch_' +
#                        str(start_epoch) + '_lr_' + str(learning_rate) +
#                        '_weights.h5')

params_model.compile(
    optimizer=adam, sample_weight_mode="temporal",
    loss={'params_output': params_loss}
)

flags_model.compile(
    optimizer=adam, sample_weight_mode="temporal",
    loss={'flags_output': flags_loss}
)

###############
# Train model #
###############
if pretrain:
    print('Pretraining' + '\n' + '-----------')

    params_training_history = []
    params_validation_history = []
    val_mcd = []
    val_pitch_rmse = []

    flags_training_history = []
    flags_validation_history = []
    val_uv_accuracy = []

    val_samples = int(len(files_list) * validation_fraction)
    sampl_epoch = int(len(files_list) - val_samples)

    batch_generator = pretrain_train_generator(
        data_path,
        batch_size=batch_size,
        basename_len=14
    )
    val_generator = pretrain_train_generator(
        data_path,
        batch_size=batch_size,
        validation=True,
        basename_len=14
    )

    for epoch in range(start_epoch, nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(np.floor(sampl_epoch / batch_size))
        progress_bar = Progbar(target=nb_batches)

        params_epoch_train_partial_loss = []
        flags_epoch_train_partial_loss = []

        try:
            progress_bar.update(0)
        except OverflowError as err:
            raise Exception('nb_batches is 0. Please check the training data')

        for index in range(nb_batches):
            train_in, train_out, train_mask = next(batch_generator)

            # TODO Separate generator data in params and flags
            params_epoch_train_partial_loss.append(
                params_model.train_on_batch(
                    {'main_input': train_in['main_input'][:, :, 0:42],
                     'src_spk_in': train_in['src_spk_in'],
                     'trg_spk_in': train_in['trg_spk_in'],
                     'feedback_in': train_in['feedback_in'][:, :, 0:42]},
                    {'params_output': train_out['params_output']},
                    sample_weight=train_mask['sample_weights'][:, :, 0]
                )
            )

            flags_epoch_train_partial_loss.append(
                flags_model.train_on_batch(
                    {'main_input': train_in['main_input'][:, :, 42:44],
                     'src_spk_in': train_in['src_spk_in'],
                     'trg_spk_in': train_in['trg_spk_in'],
                     'feedback_in': train_in['feedback_in'][:, :, 42:44]},
                    {'flags_output': train_out['flags_output']},
                    sample_weight=train_mask['sample_weights'][:, :, 0]
                )
            )

            progress_bar.update(index + 1)

        # Obtain validation losses and metrics
        (val_input, val_target, val_mask) = next(val_generator)

        params_epoch_val_loss = params_model.evaluate(
            {'main_input': val_input['main_input'][:, :, 0:42],
             'src_spk_in': val_input['src_spk_in'],
             'trg_spk_in': val_input['trg_spk_in'],
             'feedback_in': val_input['feedback_in'][:, :, 0:42]},
            {'params_output': val_target['params_output']},
            sample_weight=val_mask['sample_weights'][:, :, 0],
            batch_size=batch_size
        )

        flags_epoch_val_loss = flags_model.evaluate(
            {'main_input': val_input['main_input'][:, :, 42:44],
             'src_spk_in': val_input['src_spk_in'],
             'trg_spk_in': val_input['trg_spk_in'],
             'feedback_in': val_input['feedback_in'][:, :, 42:44]},
            {'flags_output': val_target['flags_output']},
            sample_weight=val_mask['sample_weights'][:, :, 0],
            batch_size=batch_size
        )

        # Mask data
        params_val_pred = params_model.predict_on_batch({'main_input': val_input['main_input'][:, :, 0:42],
             'src_spk_in': val_input['src_spk_in'],
             'trg_spk_in': val_input['trg_spk_in'],
             'feedback_in': val_input['feedback_in'][:, :, 0:42]})
        flags_val_pred = flags_model.predict_on_batch({'main_input': val_input['main_input'][:, :, 42:44],
             'src_spk_in': val_input['src_spk_in'],
             'trg_spk_in': val_input['trg_spk_in'],
             'feedback_in': val_input['feedback_in'][:, :, 42:44]})

        assert len(params_val_pred) == len(flags_val_pred)

        for sequence in range(len(params_val_pred)):
            params_val_pred[sequence] = np.ma.array(
                params_val_pred[sequence],
                mask=np.logical_not(np.repeat(
                    val_mask['sample_weights'][sequence],
                    params_val_pred[sequence].shape[1],
                    axis=1
                ))
            )

            flags_val_pred[sequence] = np.ma.array(
                flags_val_pred[sequence],
                mask=np.logical_not(np.repeat(
                    val_mask['sample_weights'][sequence],
                    flags_val_pred[sequence].shape[1],
                    axis=1
                ))
            )

        if py_version.major == 3:
            dict_iter = val_target.items()
        else:
            dict_iter = val_target.iteritems()

        for key, value in dict_iter:
            val_target[key] = np.ma.array(
                value,
                mask=np.logical_not(np.repeat(
                    val_mask['sample_weights'],
                    value.shape[2],
                    axis=2
                ))
            )

        # Compute validation metrics
        val_mcd.append(error_metrics.MCD(
            np.ma.compress_rows(
                val_target['params_output'][:, :, 0:40].reshape((-1, 40))),
            np.ma.compress_rows(params_val_pred[:, :, 0:40].reshape((-1, 40)))
        ))

        # val_pitch_rmse.append(error_metrics.RMSE(
        #     np.ma.compress_rows(
        #         val_target['params_output'][:, :, 40].reshape((-1, 1))),
        #     np.ma.compress_rows(params_val_pred[:, :, 40].reshape((-1, 1)))
        # )[0])

        # uv_accuracy, _, _, _ = error_metrics.AFPR(
        #     np.ma.compress_rows(
        #         np.round(val_target['flags_output'][:, :, 0]).reshape((-1, 1))),
        #     np.ma.compress_rows(np.round(flags_val_pred[:, :, 0]).reshape((-1, 1)))
        # )
        # val_uv_accuracy.append(uv_accuracy)

        params_epoch_train_loss = np.mean(np.array(params_epoch_train_partial_loss), axis=0)
        flags_epoch_train_loss = np.mean(np.array(flags_epoch_train_partial_loss), axis=0)

        params_training_history.append(params_epoch_train_loss)
        params_validation_history.append(params_epoch_val_loss)

        flags_training_history.append(flags_epoch_train_loss)
        flags_validation_history.append(flags_epoch_val_loss)

        # Generate epoch report
        print('PARAMETERS - loss: ' + str(params_training_history[-1]) +
              ' - val_loss: ' + str(params_validation_history[-1]) + '\n')
        print('FLAGS - loss: ' + str(flags_training_history[-1]) +
              ' - val_loss: ' + str(flags_validation_history[-1]) + '\n')
        print('Cepstrum MCD: ' + str(val_mcd[-1]) + ' dB')
        # print('Pitch RMSE: ' + str(val_pitch_rmse[-1]))
        # print('U/V accuracy: ' + str(val_uv_accuracy[-1]))

        ###########################
        # Save model after each epoch #
        ###########################
        print('Saving model\n' + '=' * 8 * 5)

        params_model.save_weights(
            'models/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
            '_lr_' + str(learning_rate) + '_weights.h5')

        with open('models/' + model_description + '_' + params_loss + '_' +
                  flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
                  '_lr_' + str(learning_rate) + '_model.json', 'w'
                  ) as model_json:
            model_json.write(params_model.to_json())

        flags_model.save_weights(
            'models/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
            '_lr_' + str(learning_rate) + '_weights.h5')

        with open('models/' + model_description + '_' + params_loss + '_' +
                  flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
                  '_lr_' + str(learning_rate) + '_model.json', 'w'
                  ) as model_json:
            model_json.write(flags_model.to_json())

    # Save metrics
    np.savetxt(
        'training_results/' + model_description + '_' + params_loss + '_' +
        flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) +
        '_lr_' + str(learning_rate) + '_mcd.csv', val_mcd, delimiter=','
    )
    np.savetxt(
        'training_results/' + model_description + '_' + params_loss + '_' +
        flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) +
        '_lr_' + str(learning_rate) + '_rmse.csv', val_pitch_rmse, delimiter=','
    )
    np.savetxt(
        'training_results/' + model_description + '_' + params_loss + '_' +
        flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) +
        '_lr_' + str(learning_rate) + '_acc.csv', val_uv_accuracy, delimiter=','
    )

else:
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
              ' - val_loss: ' + str(validation_history[-1]) + '\n')
        print()

        ###########################
        # Saving after each epoch #
        ###########################
        print('Saving model\n' + '=' * 8 * 5)

        model.save_weights(
            'models/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
            '_lr_' + str(learning_rate) + '_weights.h5')

        with open('models/' + model_description + '_' + params_loss + '_' +
                          flags_loss + '_' + optimizer_name + '_epoch_' + str(
            epoch) +
                          '_lr_' + str(learning_rate) + '_model.json', 'w'
                  ) as model_json:
            model_json.write(model.to_json())

print('Saving training parameters\n' + '=' * 8 * 5)
with h5py.File('training_results/' + model_description +
               '_training_params.h5', 'w') as f:
    f.attrs.create('params_loss', np.string_(params_loss))
    f.attrs.create('flags_loss', np.string_(flags_loss))
    f.attrs.create('optimizer', np.string_(optimizer_name))
    f.attrs.create('epochs', nb_epochs, dtype=int)
    f.attrs.create('learning_rate', learning_rate)
    f.attrs.create('train_speakers_max', train_speakers_max)
    f.attrs.create('train_speakers_min', train_speakers_min)
    f.attrs.create('params_metrics_names',
                   [np.string_(name) for name in params_model.metrics_names])
    f.attrs.create('flags_metrics_names',
                   [np.string_(name) for name in flags_model.metrics_names])

print('Saving training results')
np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) +
    '_lr_' + str(learning_rate) + '_epochs.csv', np.arange(nb_epochs),
    delimiter=','
)

np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_params_loss.csv', params_training_history,
    delimiter=','
)
np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_flags_loss.csv', flags_training_history,
    delimiter=','
)

np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_params_val_loss.csv', params_validation_history,
    delimiter=','
)

np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_flags_val_loss.csv', flags_validation_history,
    delimiter=','
)

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

end_time = time()
print('Elapsed time: ' + display_time(end_time - start_time))

exit()
