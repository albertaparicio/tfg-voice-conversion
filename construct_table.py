'''
Created by albert aparicio on 21/10/16

This script takes a source speech parameters file, a target file and a DTW frames file

Usage:
    construct_table.py source_param.file target_param.file dtw_frames.file
'''

import sys

# Read input files and assign them to variables
if len(sys.argv) != 3:
    print('foo')
else:
    print('Usage: construct_table.py source_param.file target_param.file dtw_frames.file')
