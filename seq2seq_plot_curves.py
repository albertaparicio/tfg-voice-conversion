# Created by Albert Aparicio on 6/12/16
# coding: utf-8

# This script takes the results of a training and plots its loss curves

import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('training_results/seq2seq_training_params.h5', 'r') as f:
    params_loss = f.attrs.get('params_loss').decode('utf-8')
    flags_loss = f.attrs.get('flags_loss').decode('utf-8')
    optimizer_name = f.attrs.get('optimizer').decode('utf-8')
    nb_epochs = f.attrs.get('epochs')
    learning_rate = f.attrs.get('learning_rate')
    metrics_names = f.attrs.get('metrics_names')

epoch = np.loadtxt('training_results/seq2seq_' + params_loss + '_' +
                   flags_loss + '_' + optimizer_name + '_epochs_' +
                   str(nb_epochs) + '_lr_' + str(learning_rate) + '_epochs.csv',
                   delimiter=',', skiprows=1)
losses = np.loadtxt('training_results/seq2seq_' + params_loss + '_' +
                    flags_loss + '_' + optimizer_name + '_epochs_' +
                    str(nb_epochs) + '_lr_' + str(learning_rate) +
                    '_loss.csv', delimiter=',', skiprows=1)
val_losses = np.loadtxt('training_results/seq2seq_' + params_loss + '_' +
                        flags_loss + '_' + optimizer_name + '_epochs_' +
                        str(nb_epochs) + '_lr_' + str(learning_rate) +
                        '_val_loss.csv', delimiter=',', skiprows=1)

assert (val_losses.size == losses.size)

# ##############################################
# # TODO Delete after dev
# metrics_names = ['loss', 'params_output_loss', 'flags_output_loss']
#
# ##############################################

plt.figure(figsize=(14, 8))
plt.plot(epoch, losses, epoch, val_losses, '--', linewidth=2)

# Prepare legend
legend_list = list(metrics_names)  # We use list() to make a copy

for name in metrics_names:
    legend_list.append('val_' + name)

plt.legend(legend_list, loc='best')
plt.grid(b='on')
plt.title('Parameters loss: ' + params_loss + ', Flags loss: ' + flags_loss +
          ', Optimizer: ' + optimizer_name + ', Epochs: ' + str(nb_epochs) +
          ', Learning rate: ' + str(learning_rate)
          )
plt.savefig('training_results/seq2seq_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epochs_' +
            str(nb_epochs) + '_lr_' + str(learning_rate) + '_graph.png',
            bbox_inches='tight')
plt.show()

exit()
