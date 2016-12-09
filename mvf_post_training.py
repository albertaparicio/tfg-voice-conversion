# Created by albert aparicio on 28/10/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import h5py
import numpy as np
from keras.models import model_from_json
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

from ahoproc_tools.error_metrics import RMSE
from tfglib.utils import apply_context

# TODO Plot loss curve

#######################
# Sizes and constants #
#######################
batch_size = 300
nb_epochs = 50
learning_rate = 0.001
context_size = 1

##############
# Load model #
##############
print('Loading model...', end='')
with open('models/mvf_model.json', 'r') as model_json:
    model = model_from_json(model_json.read())

model.load_weights('models/mvf_weights.h5')

rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mae', optimizer=rmsprop)

#############
# Load data #
#############
# Load training statistics
with h5py.File('models/mvf_train_stats.h5', 'r') as train_stats:
    src_train_mean = train_stats['src_train_mean'].value
    src_train_std = train_stats['src_train_std'].value
    trg_train_mean = train_stats['trg_train_mean'].value
    trg_train_std = train_stats['trg_train_std'].value

    train_stats.close()

# Load test data
print('Loading test data...', end='')
with h5py.File('data/test_datatable.h5', 'r') as test_datatable:
    test_data = test_datatable['test_data'][:, :]

    test_datatable.close()

src_test_data = test_data[:, 41:43]  # Source data
src_test_data[:, 0] = (src_test_data[:, 0] - src_train_mean) / src_train_std

trg_test_data = test_data[:, 84:86]  # Target data

# Apply context
src_test_data_context = np.column_stack((
    apply_context(src_test_data[:, 0], context_size), src_test_data[:, 1]
))
print('done')

################
# Predict data #
################
print('Predicting')
prediction = model.predict(src_test_data_context)

# De-normalize predicted output
prediction[:, 0] = (prediction[:, 0] * trg_train_std) + trg_train_mean

#################
# Error metrics #
#################
# Compute and print RMSE of test data
rmse_test = RMSE(
    trg_test_data[:, 0],
    prediction[:, 0],
    mask=trg_test_data[:, 1]
)

print('Test RMSE: ', rmse_test)

# Histogram of predicted training data and training data itself
plt.hist(prediction[:, 0], bins=100)
plt.title('Prediction frames')
plt.savefig('prediction_hist.png', bbox_inches='tight')
plt.show()

# # Histogram of training samples
# plt.figure()
# plt.hist(vf_gtruth, bins=100)
# plt.title('Training target frames')
# plt.savefig('gtruth_hist.png', bbox_inches='tight')
# plt.show()

exit()
