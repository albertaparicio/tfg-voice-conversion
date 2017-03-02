# Created by Albert Aparicio on 7/12/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from os.path import join

from h5py import File

data_path = 'pretrain_data/test'
params_file = 'pretrain_params.h5'

with File(join(data_path, params_file), 'r') as file:
    files_list_encoded = file['files_list'][:]
    test_files_list = [n[0] for n in files_list_encoded]

with open(join(data_path, 'pretrain_basenames.list'), 'w') as thefile:
    for item in test_files_list:
        thefile.write("%s\n" % item.decode('utf-8'))
