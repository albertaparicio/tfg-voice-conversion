# Utils module
# coding: utf-8

# Copyright © 2016 Albert Aparicio
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def kronecker_delta(x):
    if x == 0:
        return 1
    else:
        return 0


def reshape_lstm(a, tsteps, data_dim):
    """Zero-pad and reshape the input matrix 'a' so it can be fed into a stateful LSTM-based RNN"""
    # Compute the amount of zero-vectors that need to be added to the matrix
    zpad_size = int((np.ceil(a.shape[0] / tsteps) * tsteps) - a.shape[0])

    # Initialize padding matrix
    zeros_matrix = np.zeros((zpad_size, a.shape[1]))

    # Concatenate 'a' and zero-padding matrix
    a = np.concatenate((a, zeros_matrix))

    # Reshape training and test data
    return a.reshape((int(a.shape[0] / tsteps), tsteps, data_dim))
