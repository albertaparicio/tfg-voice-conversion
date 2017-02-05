# Created by albert aparicio on 04/02/2017
# coding: utf-8

# This script reads DTW files and counts the repetitions of each frame, for
# later computing the statistic distribution of the repetitions

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

from os import walk, path

import h5py
import numpy as np
from keras.utils.generic_utils import Progbar

dtw_path = 'data/training/dtw/beam2'
seen = set()

# (number of frames, number of files, 2)
distribution = np.empty((2000, 150, 2))

for root, dirs, files in walk(dtw_path):
    for index, file in enumerate(files):
        print(file)

        dtw_data = np.loadtxt(
            path.join(dtw_path, file),
            delimiter='\t',
            dtype=int
        )

        for i in range(dtw_data.shape[1]):

            for x in dtw_data[:, i]:
                if x not in seen:
                    seen.add(x)
                    distribution[x, index, i] = 1
                else:
                    distribution[x, index, i] += 1

            seen = set()

distribution = distribution.reshape((-1, 2))
mask = []

for row in range(0, distribution.shape[0]):
    if np.array_equal(distribution[row, :], [0, 0]):
        mask.append([True, True])
    else:
        mask.append([False, False])

distribution = np.ma.compress_rows(np.ma.array(distribution, mask=mask))

# plt.hist(
#     distribution[:, 0] - distribution[:, 1],
#     bins=50,
#     rwidth=.5,
#     align='left'
# )
# plt.show()

# Count repetitions of each entry in distribution
dist_list = (distribution[:, 0] - distribution[:, 1]).tolist()

values = []
probabilities = []

print('Computing probabilities of each repetition' + '\n' +
      '==========================================')

progress_bar = Progbar(target=len(dist_list))
progress_bar.update(0)

for index, item in enumerate(dist_list):
    if item not in values:
        # probabilities[str(item)] = dist_list.count(item) / len(dist_list)
        values.append(item)
        probabilities.append(dist_list.count(item) / len(dist_list))

    progress_bar.update(index + 1)

print('Saving probabilities\n' + '====================')
with h5py.File('pretrain_data/dtw_probabilities.h5', 'w') as f:
    # Save numbers and probabilities
    f.create_dataset(
        'values',
        data=np.array(values, dtype=int),
        compression="gzip",
        compression_opts=9
    )
    f.create_dataset(
        'probabilities',
        data=np.array(probabilities),
        compression="gzip",
        compression_opts=9
    )

    f.close()

exit()
