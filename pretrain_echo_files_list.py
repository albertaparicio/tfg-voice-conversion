# Created by Albert Aparicio on 7/12/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function
from tfglib.pretrain_data_params import pretrain_load_data_parameters
from os.path import join

data_path = 'pretrain_data/test'

(_,_,_,test_files_list) = pretrain_load_data_parameters(data_path)

with open(join(data_path, 'basenames.list'), 'w') as thefile:
    for item in test_files_list:
        thefile.write("%s\n" % item.decode('utf-8'))


# [string.decode('utf-8') for string in test_files_list]
