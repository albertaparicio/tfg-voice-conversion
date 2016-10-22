# Created by albert aparicio on 21/10/16

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from collections import namedtuple

import numpy as np


def read_params(param_len, source_path, target_path, dtw_path):
    """This function takes as its inputs:

        param_len: The length of the parameters vector of each vocoder sound file
        source_path: The file path of the source data (in ASCII, and each file's parameters are located in a single row)
        target_path: The file path of the target data (in ASCII, and each file's parameters are located in a single row)
        dtw_path: The file path of the DTW frame matchings (in 2 ASCII columns, the left one being the source
                    and the right one, the target)

        The outputs are:

        source_params: A NumPy's 'ndarray' with the source's parameters organized by rows.
                        Its size is len(source_path) -i.e. the number of source files- x param_len.
        target_params: A NumPy's 'ndarray' with the target's parameters organized by rows.
                        Its size is len(target_path) -i.e. the number of target files- x param_len.
        dtw_frames:    A NumPy's 'ndarray' with the DTW matchings organized by rows.
                        Its size is len(source_path) (i.e. the number of source files) x 2."""

    # Source
    source_file = open(source_path, 'r')
    source_lines = source_file.readlines()

    source_params = np.empty([len(source_lines), param_len])
    for index in range(0, len(source_lines) - 1, 1):
        aux = source_lines[index].split('\n')
        source_params[index, :] = aux[0].split('\t')

    del source_file
    del source_lines

    # Target
    target_file = open(target_path, 'r')
    target_lines = target_file.readlines()

    target_params = np.empty([len(target_lines), param_len])
    for index in range(0, len(target_lines) - 1, 1):
        aux = target_lines[index].split('\n')
        target_params[index, :] = aux[0].split('\t')

    del target_file
    del target_lines

    # DTW Frames
    frames_file = open(dtw_path, 'r')
    frames_lines = frames_file.readlines()

    dtw_frames = np.empty([len(frames_lines), 2])
    for index in range(0, len(frames_lines) - 1, 1):
        aux = frames_lines[index].split('\n')
        dtw_frames[index, :] = aux[0].split('\t')

    del frames_file
    del frames_lines

    # Organize output variables in a tuple
    ret_params = namedtuple('Params', ['dtw', 'source', 'target'])
    return ret_params(dtw=dtw_frames, source=source_params, target=target_params)


def align_frames(dtw_frames, source_params, target_params):
    """This functions aligns the source and target frames according to the rows of 'dtw_frames'.

    The inputs of this function are the outputs of the 'read_params'function."""

    assert source_params.shape[1] == target_params.shape[1]

    data = np.empty([dtw_frames.shape[0], source_params.shape[1] + target_params.shape[1]])
    for row, matching in enumerate(dtw_frames):
        data[row, :] = np.concatenate((
            source_params[int(dtw_frames[row, 0]), :],
            target_params[int(dtw_frames[row, 1]), :]
        ))

    return data


# params = read_params(1, 'vocoded/test_source/SF1_TF1_200001.lf0.dat', 'vocoded/test_target/SF1_TF1_200001.lf0.dat',
#                 'frames/SF1_TF1_200001.frames.txt')

params = read_params(40, 'vocoded/test_source/SF1_TF1_200001.mcp.dat', 'vocoded/test_target/SF1_TF1_200001.mcp.dat',
                     'frames/SF1_TF1_200001.frames.txt')

data_table = align_frames(params[0], params[1], params[2])

np.savetxt('datatable.csv', data_table, delimiter=',')
print('done')
# params = read_params(1, 'vocoded/test_source/SF1_TF1_200001.vf.dat', 'vocoded/test_target/SF1_TF1_200001.vf.dat',
#                 'frames/SF1_TF1_200001.frames.txt')
