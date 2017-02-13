# Created by albert aparicio on 21/11/16
# coding: utf-8

# This script defines an encoder-decoder GRU-RNN model
# to map the source's sequence parameters to the target's sequence parameters

# TODO Document and explain steps

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from time import time

import h5py
import numpy as np
import tfglib.seq2seq_datatable as s2s
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.layers import Input, Dropout, Dense, merge, TimeDistributed
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
from tfglib.pretrain_data_params import pretrain_load_data_parameters
from tfglib.pretrain_data_params import pretrain_save_data_parameters
from tfglib.pretrain_data_params import pretrain_train_generator
from tfglib.seq2seq_normalize import maxmin_scaling
from tfglib.utils import display_time
from tfglib.utils import reverse_encoder_output, reversed_output_shape

# Save training start time
start_time = time()

############
# Switches #
############
pretrain = True

# Decide if datatable/parameters must be built or can be loaded from a file
build_datatable = False

#######################
# Sizes and constants #
#######################
model_description = 'seq2seq_pretrain'

# Batch shape
batch_size = 10
output_dim = 44
data_dim = output_dim + 10 + 10
emb_size = 256

# Other constants
nb_epochs = 20
learning_rate = 0.001
validation_fraction = 0.25

#############
# Load data #
#############
if pretrain:
    data_path = 'pretrain_data'

    if build_datatable:
        print('Saving pretraining parameters')
        (
            max_train_length,
            spk_max,
            spk_min,
            files_list
        ) = pretrain_save_data_parameters(data_path)

    else:
        print('Load pretraining parameters')
        (
            max_train_length,
            spk_max,
            spk_min,
            files_list
        ) = pretrain_load_data_parameters(data_path)

    train_speakers = spk_max.shape[0]

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
main_input = Input(shape=(max_train_length, output_dim),
                   dtype='float32',
                   name='main_input')

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
    input_length=max_train_length
)

merged_parameters = merge(
    [main_input,
     embedded_spk_indexes(src_spk_input),
     embedded_spk_indexes(trg_spk_input)
     ],
    mode='concat'
)

encoder_PLSTM = PLSTM(
    output_dim=emb_size,
    return_sequences=True,
    consume_less='gpu',
)(merged_parameters)

reversed_encoder = Lambda(
    reverse_encoder_output,
    output_shape=reversed_output_shape
)(encoder_PLSTM)

# Feedback input
feedback_in = Input(shape=(max_train_length, output_dim), name='feedback_in')
dec_in = merge([reversed_encoder, feedback_in], mode='concat')

decoder_PLSTM = PLSTM(
    emb_size,
    return_sequences=True,
    consume_less='gpu'
)(dec_in)

dropout_layer = Dropout(0.5)(decoder_PLSTM)

parameters_PLSTM = PLSTM(
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

model = Model(input=[main_input, src_spk_input, trg_spk_input, feedback_in],
              output=[parameters_PLSTM, flags_Dense])

optimizer_name = 'adam'
adam = Adam(clipnorm=5)
params_loss = 'mse'
flags_loss = 'mse'

model.compile(
    optimizer=adam, sample_weight_mode="temporal",
    loss={'params_output': params_loss, 'flags_output': flags_loss}
)

###############
# Train model #
###############
if pretrain:
    print('Pretraining' + '\n' + '-----------')

    val_samples = int(np.floor(len(files_list) * validation_fraction))
    sampl_epoch = int(len(files_list) - val_samples)

    checkpointer = ModelCheckpoint(
        filepath='models/' + model_description + '_' + params_loss + '_' +
                 flags_loss + '_' + optimizer_name + '_epoch_' + '{epoch:02d}' +
                 '_lr_' + str(learning_rate) + '_model.h5',
        verbose=1
    )

    # TODO Canviar fit_generator per un train_on_batch
    # Iterar per batches - dos variables
    # for x,y_true,mask in pretrain_train_generator
    #   model.train_on_batch
    #   y_pred = model.predict(mateix batch en què he entrenat)
    #   Reshape per a posar-ho en 2D, i aplicar mètriques del Santi
    history = model.fit_generator(
        pretrain_train_generator(
            data_path,
            batch_size=batch_size
        ),
        samples_per_epoch=sampl_epoch,
        nb_epoch=nb_epochs,
        validation_data=pretrain_train_generator(
            data_path,
            batch_size=batch_size,
            validation=True
        ),
        nb_val_samples=val_samples,
        callbacks=[checkpointer]
    )

    epochs = history.epoch
    training_history = history.history['loss']
    validation_history = history.history['val_loss']

    print('Saving training results')
    np.savetxt(
        'training_results/' + model_description + '_' + params_loss + '_' +
        flags_loss + '_' + optimizer_name + '_epochs_' + str(
            nb_epochs) + '_lr_' +
        str(learning_rate) + '_epochs.csv', epochs, delimiter=','
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
              ' - val_loss: ' + str(validation_history[-1]) +
              '\n'  # + '-' * 24
              )
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
                  flags_loss + '_' + optimizer_name + '_epoch_' + str(epoch) +
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
        f.attrs.create('metrics_names',
                       [np.string_(name) for name in model.metrics_names]
                       )

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
    str(learning_rate) + '_loss.csv', training_history, delimiter=','
)
np.savetxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_val_loss.csv', validation_history, delimiter=','
)

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

end_time = time()
print('Elapsed time: ' + display_time(end_time - start_time))

exit()
