# Created by albert aparicio on 02/04/17
# coding: utf-8

# This script defines an Sequence-to-Sequence model, using the implemented
# encoders and decoders from TensorFlow

# TODO Document and explain steps

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import logging

import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)