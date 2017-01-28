# Created by Albert Aparicio on 9/1/17
# coding: utf-8

# This script takes the predictions of the U/V flag and plots its ROC curves

import matplotlib.pyplot as plt
import numpy as np
import tfglib.seq2seq_datatable as s2s
from sklearn.metrics import roc_curve, auc
from tfglib.construct_table import parse_file

# Load test database
print('Loading test datatable...', end='')
(src_test_datatable,
 src_test_masks,
 trg_test_datatable,
 trg_test_masks,
 max_test_length,
 test_speakers_max,
 test_speakers_min
 ) = s2s.seq2seq2_load_datatable(
    'data/seq2seq_test_datatable.h5'
)
print('done')

##################
# Load basenames #
##################
basenames_file = open('data/test/seq2seq_basenames.list', 'r')
basenames_lines = basenames_file.readlines()
# Strip '\n' characters
basenames = [line.split('\n')[0] for line in basenames_lines]

# Load speakers
speakers_file = open('data/test/speakers.list', 'r')
speakers_lines = speakers_file.readlines()
# Strip '\n' characters
speakers = [line.split('\n')[0] for line in speakers_lines]

#######################
# Loop over sequences #
#######################
assert len(basenames) == src_test_datatable.shape[0] / np.square(len(speakers))

# Preallocate False-Positive and True-Positive matrices
fpr = dict()
tpr = dict()
roc_auc = dict()
src_spk_ind = 0
trg_spk_ind = 0

for src_spk in speakers:
    fpr[src_spk] = dict()
    tpr[src_spk] = dict()
    roc_auc[src_spk] = dict()

    for trg_spk in speakers:
        fpr[src_spk][trg_spk] = dict()
        tpr[src_spk][trg_spk] = dict()
        roc_auc[src_spk][trg_spk] = dict()

        # for i in range(src_test_datatable.shape[0]):
        for i in range(len(basenames)):
            # TODO Consider plotting an averaged ROC for each spk combination
            print(src_spk + '->' + trg_spk + ' ' + basenames[i])
            # TODO figure out if this is necessary
            # fpr[src_spk][trg_spk][basenames[i]] = dict()
            # tpr[src_spk][trg_spk][basenames[i]] = dict()
            # roc_auc[src_spk][trg_spk][basenames[i]] = dict()

            # Load raw U/V flags
            raw_uv = parse_file(1,
                                'data/test/s2s_predicted/' + src_spk + '-' +
                                trg_spk + '/' + basenames[i] + '.uv.dat')

            # Round U/V flags
            rounded_uv = np.round(raw_uv)

            # Compute ROC curve and the area under it
            (
                fpr[src_spk][trg_spk][basenames[i]],
                tpr[src_spk][trg_spk][basenames[i]],
                _
            ) = roc_curve(
                trg_test_datatable[
                    i + (src_spk_ind + trg_spk_ind) * len(basenames), :, 42
                ],
                rounded_uv
            )

            roc_auc[src_spk][trg_spk][basenames[i]] = auc(
                fpr[src_spk][trg_spk][basenames[i]],
                tpr[src_spk][trg_spk][basenames[i]]
            )

            # Plot and save ROC curve
            fig = plt.figure()
            lw = 2
            plt.plot(fpr[src_spk][trg_spk][basenames[i]],
                     tpr[src_spk][trg_spk][basenames[i]],
                     color='darkorange',
                     lw=lw,
                     label='ROC curve (area = %0.2f)' %
                           roc_auc[src_spk][trg_spk][basenames[i]])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig(
                'training_results/uv_roc_' + src_spk + '-' + trg_spk +
                '-' + basenames[i] + '.png',
                bbox_inches='tight')
            # plt.show()
            # plt.close("all")

        trg_spk_ind += 1
    src_spk_ind += 1
