# Created by Albert Aparicio on 21/10/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
import h5py

from keras.models import model_from_json
from keras.optimizers import RMSprop

from construct_table import parse_file
import utils

###############
# Load models #
###############
# mvf model
###########
mvf_lr = 0.001

with open('mvf_model.json', 'r') as model_json:
    mvf_model = model_from_json(model_json.read())

mvf_model.load_weights('mvf_weights.h5')

mvf_rmsprop = RMSprop(lr=mvf_lr)
mvf_model.compile(loss='mae', optimizer=mvf_rmsprop)

# Load training statistics
with h5py.File('mvf_train_stats.h5', 'r') as train_stats:
    src_mvf_mean = train_stats['src_train_mean'].value
    src_mvf_std = train_stats['src_train_std'].value
    trg_mvf_mean = train_stats['trg_train_mean'].value
    trg_mvf_std = train_stats['trg_train_std'].value

    train_stats.close()

# log(f0) model
###############
# Batch shape
lf0_batch_size = 1
lf0_tsteps = 50
lf0_data_dim = 2

with open('lf0_model.json', 'r') as model_json:
    lf0_model = model_from_json(model_json.read())

lf0_model.load_weights('lf0_weights.h5')
lf0_model.compile(loss='mse', optimizer='rmsprop')

# Load training statistics
with h5py.File('lf0_train_stats.h5', 'r') as train_stats:
    src_lf0_mean = train_stats['src_train_mean'].value
    src_lf0_std = train_stats['src_train_std'].value
    trg_lf0_mean = train_stats['trg_train_mean'].value
    trg_lf0_std = train_stats['trg_train_std'].value

    train_stats.close()

# cepstrum parameters model
############################
mcp_lr = 0.0001
# Batch shape
mcp_batch_size = 1
mcp_tsteps = 50
mcp_data_dim = 40

with open('mcp_model.json', 'r') as model_json:
    mcp_model = model_from_json(model_json.read())

mcp_model.load_weights('mcp_weights.h5')

mcp_rmsprop = RMSprop(lr=mcp_lr)
mcp_model.compile(loss='mse', optimizer=mcp_rmsprop)

# Load training statistics
with h5py.File('mcp_train_stats.h5', 'r') as train_stats:
    src_mcp_mean = train_stats['src_train_mean'][:]
    src_mcp_std = train_stats['src_train_std'][:]
    trg_mcp_mean = train_stats['trg_train_mean'][:]
    trg_mcp_std = train_stats['trg_train_std'][:]

    train_stats.close()

##################
# Load basenames #
##################
basenames_file = open('data/test/basenames.list', 'r')
basenames_lines = basenames_file.readlines()

# Strip '\n' characters
basenames = [line.split('\n')[0] for line in basenames_lines]

###################
# Loop over files #
###################
for basename in basenames:
    ###################
    # Load parameters #
    ###################
    mcp_params = parse_file(40,
                            'data/test/vocoded/SF1/' + basename + '.mcp.dat'
                            )
    lf0_params = parse_file(1,
                            'data/test/vocoded/SF1/' + basename + '.lf0.i.dat'
                            )
    mvf_params = parse_file(1,
                            'data/test/vocoded/SF1/' + basename + '.vf.i.dat'
                            )

    # Compute U/V flags
    assert mvf_params.shape == lf0_params.shape
    uv_flags = np.empty(mvf_params.shape)
    for index, vf in enumerate(uv_flags):
        uv_flags[index] = 1 - utils.kronecker_delta(mvf_params[index])

    # Prepare data for prediction
    mcp_params = (mcp_params - src_mcp_mean) / src_mcp_std
    mcp_params = utils.reshape_lstm(mcp_params, mcp_tsteps, mcp_data_dim)

    lf0_params = (lf0_params - src_lf0_mean) / src_lf0_std
    lf0_params = utils.reshape_lstm(lf0_params, lf0_tsteps, lf0_data_dim)

    mvf_params = (mvf_params - src_mvf_mean) / src_mvf_std

    ######################
    # Predict parameters #
    ######################
    mvf_prediction = mvf_model.predict(np.column_stack(mvf_params, uv_flags))
    mvf_prediction[:, 0] = (mvf_prediction[:, 0] * trg_mvf_std) + trg_mvf_mean

    lf0_prediction = lf0_model.predict(lf0_params, batch_size=lf0_batch_size)
    lf0_prediction = lf0_prediction.reshape(-1, 2)
    lf0_prediction[:, 0] = (lf0_prediction[:, 0] * trg_lf0_std) + trg_lf0_mean

    mcp_prediction = mcp_model.predict(mcp_params, batch_size=mcp_batch_size)
    mcp_prediction = mcp_prediction.reshape(-1, mcp_data_dim)
    mcp_prediction = (mcp_prediction * trg_mcp_std) + trg_mcp_mean

    # Apply U/V flag to mvf and lf0 data
    for index, entry in enumerate(lf0_prediction):
        if entry[1] == 0:
            lf0_prediction[index, 0] = -1e+10
            mvf_prediction[index, 0] = 0

    ###########################
    # Save parameters to file #
    ###########################
    np.savetxt(
        'data/test/predicted/SF1-TF1/' + basename + '.vf.dat',
        mvf_prediction[:, 0]
    )
    np.savetxt(
        'data/test/predicted/SF1-TF1/' + basename + '.lf0.dat',
        lf0_prediction[:, 0]
    )
    np.savetxt(
        'data/test/predicted/SF1-TF1/' + basename + '.mcp.dat',
        mcp_prediction,
        delimiter='\t'
    )

    #####################
    # Decode parameters #
    #####################
    # TODO call ahodecoder16_64
    import os
    f = os.popen(
        "bash decode_aho.sh 'data/test/predicted/SF1-TF1/' " + basename
    )
