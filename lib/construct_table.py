# Created by Albert Aparicio on 21/10/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from tfglib.utils import kronecker_delta


def parse_file(param_len, file_path, offset=0):
    """This function parses a vocoder data file.

    INPUTS:

        param_len: The length of the parameters vector of each vocoder frame
        file_path: The file path of the data
                    (each frame's parameters are located in a single row)

    OUTPUTS:

        file_params: NumPy.ndarray with the file's parameters organized by rows
                     size is len(source_path) x param_len."""

    # Source
    open_file = open(file_path, 'r')
    file_lines = open_file.readlines()

    # Preallocate matrix for an increased memory efficiency
    file_params = np.empty([len(file_lines) - offset, param_len])
    for index in range(offset, len(file_lines), 1):
        aux = file_lines[index].split('\n')
        file_params[index - offset, :] = aux[0].split('\t')

    return file_params


def align_frames(dtw_frames, source_params, target_params):
    """Align source and target frames according to the rows of 'dtw_frames'.

    INPUTS:
        dtw_frames: 2-column 'ndarray' with the DTW matching for each frame
        source_params: 'ndarray' with concatenated parameters of the source file
        target_params: 'ndarray' with concatenated parameters of the target file

    OUTPUT:
        An 'ndarray' with the aligned data.

    NOTE: Length of aligned array may be greater than source and target files"""

    assert source_params.shape[1] == target_params.shape[1]

    data = np.empty(
        [dtw_frames.shape[0], source_params.shape[1] + target_params.shape[1]]
    )
    for row, matching in enumerate(dtw_frames):
        data[row, :] = np.concatenate((
            source_params[int(dtw_frames[row, 0]), :],
            target_params[int(dtw_frames[row, 1]), :]
        ))

    return data


def build_file_table(basename, source_dir, target_dir, dtw_dir):
    """This function builds a datatable from the aligned vocoder frames.
    It reads and parses the input files, concatenates the data and aligns it.

    INPUTS:
        basename: the name without extension of the file to be prepared
        source_dir: directory path to the source files. (it must end in '/')
        target_fir: directory path to the target files. (it must end in '/')
        dtw_dir: dir path to DTW frame matchings file. (it must end in '/')"""

    # Parse files
    source_f0 = parse_file(1, source_dir + basename + '.' + 'lf0' + '.dat')
    source_f0_i = parse_file(
        1,
        source_dir + basename + '.' + 'lf0' + '.i.dat'
    )  # Interpolated data

    source_mcp = parse_file(40, source_dir + basename + '.' + 'mcp' + '.dat')

    source_vf = parse_file(1, source_dir + basename + '.' + 'vf' + '.dat')
    source_vf_i = parse_file(
        1,
        source_dir + basename + '.' + 'vf' + '.i.dat'
    )  # Use interpolated data

    target_f0 = parse_file(1, target_dir + basename + '.' + 'lf0' + '.dat')
    target_f0_i = parse_file(
        1,
        target_dir + basename + '.' + 'lf0' + '.i.dat'
    )  # Use interpolated data

    target_mcp = parse_file(40, target_dir + basename + '.' + 'mcp' + '.dat')

    target_vf = parse_file(1, target_dir + basename + '.' + 'vf' + '.dat')
    target_vf_i = parse_file(
        1,
        target_dir + basename + '.' + 'vf' + '.i.dat'
    )  # Use interpolated data

    dtw_frames = parse_file(2, dtw_dir + basename + '.dtw', 5)

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
        source_f0_i,
        source_vf_i,
        source_voiced
    ), axis=1)

    target_params = np.concatenate((
        target_mcp,
        target_f0_i,
        target_vf_i,
        target_voiced
    ), axis=1)

    # Align parameters
    return align_frames(dtw_frames, source_params, target_params)


def construct_datatable(basenames_list, source_dir, target_dir, dtw_dir):
    # Parse basenames list
    basenames_file = open(basenames_list, 'r')
    basenames_lines = basenames_file.readlines()

    # Strip '\n' characters
    basenames_lines = [line.split('\n')[0] for line in basenames_lines]

    # Construct table of 1st file
    datatable = build_file_table(
        basenames_lines[0],
        source_dir,
        target_dir,
        dtw_dir
    )

    # Iterate through the rest of files and concatenate them below
    for i in range(1, len(basenames_lines), 1):
        datatable = np.concatenate((
            datatable,
            build_file_table(
                basenames_lines[i],
                source_dir,
                target_dir,
                dtw_dir
            )
        ))

    # Return result
    return datatable


def save_datatable(data_dir, dataset_name, datatable_out_file):
    """This function constructs and saves the datatable in an .h5 file

    INPUTS:
        data_dir: directory of data to be used in the datatable.
                    This path must end in a '/'
        dataset_name: name of the dataset in the .h5 file
        datatable_out_file: path to the output .h5 file (no extension)

    OUTPUTS: None. The only output is the .h5 file of the datatable"""
    # Construct datatable
    data = construct_datatable(
        data_dir + 'basenames.list',
        data_dir + 'vocoded/SF1/',
        data_dir + 'vocoded/TF1/',
        data_dir + 'dtw/beam2/'
    )

    # Save and compress with gzip to save space
    with h5py.File(datatable_out_file + '.h5', 'w') as f:
        f.create_dataset(
            dataset_name,
            data=data,
            compression="gzip",
            compression_opts=9
        )

        f.close()

    return data


def load_datatable(datatable_file, dataset_name):
    """This function loads a datatable from a previously saved file

    INPUTS:
        datatable_file: path to the datatable .h5 file
        dataset_name: name of the dataset to retrieve from the .h5 file

    OUTPUT: a NumPy.ndarray with the datatable"""
    with h5py.File(datatable_file, 'r') as file:
        dataset = file[dataset_name][:, :]

        file.close()

    return dataset
