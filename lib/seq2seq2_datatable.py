# Created by Albert Aparicio on 21/10/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from tfglib.utils import kronecker_delta
from tfglib.construct_table import parse_file

from keras.utils.np_utils import to_categorical


def find_longest_sequence(data_dir, speakers_list, basenames_list):
    longest_sequence = 0

    for speaker in speakers_list:
        for basename in basenames_list:
            params = parse_file(
                1,
                data_dir + 'vocoded_s2s/' + speaker + '/' + basename + '.' + 'lf0' + '.dat'
            )

            if params.shape[0] > longest_sequence:
                longest_sequence = params.shape[0]

    return longest_sequence


def seq2seq_build_file_table(source_dir, src_index, target_dir, trg_index, basename, longest_seq):
    # Re-use code from tfglib.construct_table.build_file_table
    # Maybe add the src and trg speakers indexes, for the one-hot encoding

    # Parse parameter files
    source_mcp = parse_file(40, source_dir + basename + '.' + 'mcp' + '.dat')

    source_f0 = parse_file(1, source_dir + basename + '.' + 'lf0' + '.dat')
    source_f0_i = parse_file(
        1,
        source_dir + basename + '.' + 'lf0' + '.i.dat'
    )  # Interpolated data

    source_vf = parse_file(1, source_dir + basename + '.' + 'vf' + '.dat')
    source_vf_i = parse_file(
        1,
        source_dir + basename + '.' + 'vf' + '.i.dat'
    )  # Use interpolated data

    target_mcp = parse_file(40, target_dir + basename + '.' + 'mcp' + '.dat')

    target_f0 = parse_file(1, target_dir + basename + '.' + 'lf0' + '.dat')
    target_f0_i = parse_file(
        1,
        target_dir + basename + '.' + 'lf0' + '.i.dat'
    )  # Use interpolated data

    target_vf = parse_file(1, target_dir + basename + '.' + 'vf' + '.dat')
    target_vf_i = parse_file(
        1,
        target_dir + basename + '.' + 'vf' + '.i.dat'
    )  # Use interpolated data

    # Build voiced/unvoiced flag arrays
    # The flags are:
    #   1 -> voiced
    #   0 -> unvoiced
    assert source_vf.shape == source_f0.shape
    source_voiced = np.empty(source_vf.shape)
    for index, vf in enumerate(source_vf):
        source_voiced[index] = 1 - kronecker_delta(source_vf[index])

    assert target_vf.shape == target_f0.shape
    target_voiced = np.empty(target_vf.shape)
    for index, vf in enumerate(target_vf):
        target_voiced[index] = 1 - kronecker_delta(target_vf[index])

    # Initialize End-Of-Sequence flag
    src_eos_flag = np.zeros(source_vf.shape)
    src_eos_flag[src_eos_flag.shape[0] - 1, :] = 1

    trg_eos_flag = np.zeros(target_vf.shape)
    trg_eos_flag[trg_eos_flag.shape[0] - 1, :] = 1

    # Initialize one-hot-encoded speaker indexes
    src_spk_index = to_categorical(
        src_index * np.ones((source_vf.shape[0],)),
        nb_classes=10
    )
    trg_spk_index = to_categorical(
        trg_index * np.ones((target_vf.shape[0],)),
        nb_classes=10
    )

    # Concatenate source and target params
    source_params = np.concatenate((
        source_mcp,
        source_f0_i,
        source_vf_i,
        source_voiced,
        src_eos_flag,
        src_spk_index,
        trg_spk_index
    ), axis=1)
    # Apply zero-padding
    source_params_zp = np.concatenate((
        np.zeros((
            longest_seq - source_params.shape[0],
            source_params.shape[1]
        )),
        source_params
    ))

    target_params = np.concatenate((
        target_mcp,
        target_f0_i,
        target_vf_i,
        target_voiced,
        trg_eos_flag
    ), axis=1)
    # Apply zero-padding
    target_params_zp = np.concatenate((
        target_params,
        np.zeros((
            longest_seq - target_params.shape[0],
            target_params.shape[1]
        ))
    ))

    # Initialize padding masks, to be passed into keras' fit
    # Source mask
    source_mask = np.concatenate((
        np.zeros((
            longest_seq - source_params.shape[0],
            1
        )),
        np.ones((
            source_params.shape[0],
            1
        ))
    ))

    # Target mask
    target_mask = np.concatenate((
        np.ones((
            target_params.shape[0],
            1
        )),
        np.zeros((
            longest_seq - target_params.shape[0],
            1
        ))
    ))

    assert source_mask.shape == target_mask.shape

    return source_params_zp, source_mask, target_params_zp, target_mask


def seq2seq_construct_datatable(data_dir, speakers_file, basenames_file):
    # Parse speakers file
    speakers = open(data_dir + speakers_file, 'r').readlines()
    # Strip '\n' characters
    speakers = [line.split('\n')[0] for line in speakers]

    # Parse basenames file
    # This file should be equal for all speakers
    basenames = open(data_dir + basenames_file, 'r').readlines()
    # Strip '\n' characters
    basenames = [line.split('\n')[0] for line in basenames]

    # TODO Remove hardcoded value when definitive dataset is obtained
    longest_seq = 700
    # longest_seq = find_longest_sequence(data_dir, speakers, basenames)

    # Initialize datatable
    src_datatable = []
    src_masks = []
    trg_datatable = []
    trg_masks = []

    # Nest iterate over speakers
    for src_index, src_spk in enumerate(speakers):
        for trg_index, trg_spk in enumerate(speakers):
            for basename in basenames:
                aux_src_params, aux_src_mask, aux_trg_params, aux_trg_mask = seq2seq_build_file_table(
                    data_dir + 'vocoded_s2s/' + src_spk + '/',
                    src_index,
                    data_dir + 'vocoded_s2s/' + trg_spk + '/',
                    trg_index,
                    basename,
                    longest_seq
                )

                # Append sequence parameters and masks to the main datatables and masks
                src_datatable.append(aux_src_params)
                src_masks.append(aux_src_mask)
                trg_datatable.append(aux_trg_params)
                trg_masks.append(aux_trg_mask)

    return np.array(src_datatable), np.array(src_masks), np.array(trg_datatable), np.array(trg_masks)

# TODO implement the save_datatable and load_datatable equivalents, with .h5 file saving

# source_datatable, source_masks, target_datatable, target_masks = seq2seq_construct_datatable(
#     '../data/training/',
#     'speakers.list',
#     'seq2seq_basenames.list'
# )
#
# exit()
