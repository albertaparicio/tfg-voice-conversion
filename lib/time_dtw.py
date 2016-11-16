# Created by albert aparicio on 14/11/16
# coding: utf-8

# This script takes a .frames.txt file from SPTK's dtw output and converts
# it to time frames (5 ms frames)
#
# This script expects the path to the .frames.txt file to be passed as an
# argument

import numpy as np
from tfglib.construct_table import parse_file
from sys import argv

# Check that a file parameter has been passed
assert len(argv) == 2

# Parse frames and convert them into time
frames = 0.005 * parse_file(2, argv[1])

# Split the filename at each '.'
sp = argv[1].split('.')

# Output the resulting frames
# np.savetxt(sp[0]+'.frames.dtw', frames, fmt='%.3e', delimiter='\t', header='# REF\tTST\n# ----\t----')
np.savetxt(sp[0]+'.frames.dtw', frames, fmt='%.3f', delimiter='\t', header='REF\tTST\n----\t----')
# exit()

