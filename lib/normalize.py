# coding: utf-8
# Created by albert aparicio on 16/11/16

# Normalize mcp coefficients by their mean and variance


from __future__ import print_function

from sys import argv, exit

import numpy as np
from tfglib.construct_table import parse_file

# Define usage constant
usage = 'Usage: normalize.py [file_directory] [filename]' + \
        '[number of parameters]'

if len(argv) == 1 or \
        (len(argv) == 2 and (argv[1] == '-h' or argv[1] == '--help')):
    print(usage + "\n\nThe file directory must end in a '/'.")

elif len(argv) == 4:
    # Parse input file
    data = parse_file(int(argv[3]), argv[1] + argv[2])

    # Normalize data
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    normalized_data = (data - data_mean) / data_std

    # Output normalized data to file
    np.savetxt(argv[1] + argv[2] + '.norm', normalized_data[:, 1:normalized_data.shape[1]], fmt='%.18f', delimiter='\t')

else:
    exit('Please, input two arguments as indicated in the usage.\n\n' + usage)
