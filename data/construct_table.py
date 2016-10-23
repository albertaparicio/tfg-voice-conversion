# Created by albert aparicio on 21/10/16

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np

from utils import kronecker_delta


def parse_file(param_len, file_path):
    """This function parses a vocoder data file.

    INPUTS:

        param_len: The length of the parameters vector of each vocoder frame
        file_path: The file path of the data (each frame's parameters are located in a single row)

    OUTPUTS:

        file_params: A NumPy's 'ndarray' with the file's parameters organized by rows.
                        Its size is len(source_path) -i.e. the number of source files- x param_len."""

    # Source
    open_file = open(file_path, 'r')
    file_lines = open_file.readlines()

    file_params = np.empty([len(file_lines), param_len])
    for index in range(0, len(file_lines) - 1, 1):
        aux = file_lines[index].split('\n')
        file_params[index, :] = aux[0].split('\t')

    return file_params


def align_frames(dtw_frames, source_params, target_params):
    """This functions aligns the source and target frames according to the rows of 'dtw_frames'.

    INPUTS:
        dtw_frames: 2-column 'ndarray' with the DTW matching for each frame
        source_params: 'ndarray' with the concatenated parameters of the source file
        target_params: 'ndarray' with the concatenated parameters of the target file

    OUTPUT:
        An 'ndarray' with the aligned data.

    NOTE: The length of the aligned array may be greater than both source and target files"""

    assert source_params.shape[1] == target_params.shape[1]

    data = np.empty([dtw_frames.shape[0], source_params.shape[1] + target_params.shape[1]])
    for row, matching in enumerate(dtw_frames):
        data[row, :] = np.concatenate((
            source_params[int(dtw_frames[row, 0]), :],
            target_params[int(dtw_frames[row, 1]), :]
        ))

    return data


def build_datatable(basename, source_dir, target_dir, dtw_dir):
    """This function builds a datatable from the aligned vocoder frames.
    It reads and parses the input files, concatenates the data and aligns it.

    INPUTS:
        basename: the name without extension of the file to be prepared
        source_dir: directory path to the source files. (it must end in '/')
        target_fir: directory path to the target files. (it must end in '/')
        dtw_dir: directory path to the DTW frame matchings file. (it must end in '/')"""

    # Parse files
    source_f0 = parse_file(1, source_dir + basename + '.' + 'lf0' + '.dat')
    source_mcp = parse_file(40, source_dir + basename + '.' + 'mcp' + '.dat')
    source_vf = parse_file(1, source_dir + basename + '.' + 'vf' + '.dat')

    target_f0 = parse_file(1, target_dir + basename + '.' + 'lf0' + '.dat')
    target_mcp = parse_file(40, target_dir + basename + '.' + 'mcp' + '.dat')
    target_vf = parse_file(1, target_dir + basename + '.' + 'vf' + '.dat')

    dtw_frames = parse_file(2, dtw_dir + basename + '.frames.txt')

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

    # Concatenate source and target params
    source_params = np.concatenate((
        source_mcp,
        source_f0,
        source_vf,
        source_voiced
    ), axis=1)

    target_params = np.concatenate((
        target_mcp,
        target_f0,
        target_vf,
        target_voiced
    ), axis=1)

    # Align parameters
    return align_frames(dtw_frames, source_params, target_params)

# file_datatable = build_datatable('SF1_TF1_200001', 'vocoded/test_source/', 'vocoded/test_target/', 'frames/')
# np.savetxt('datatable.csv', file_datatable, delimiter=',')
