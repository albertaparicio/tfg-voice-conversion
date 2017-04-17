# Created by Albert Aparicio on 6/12/16
# coding: utf-8

# This script takes the results of a training and plots its loss curves

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

results_dir = 'training_results/tf_train'
batch_size = 20
epoch = 30
batches_per_epoch = 1573
total_batches = 47190

for root, dirs, files in os.walk(results_dir):
  tr_files = [file for file in sorted(files) if 'tr_losses.csv' in file]
  va_files = [file for file in sorted(files) if 'val_losses.csv' in file]
  te_files = [file for file in sorted(files) if 'te_losses.csv' in file]

  tr_batches = np.loadtxt(os.path.join(results_dir, tr_files[-1]))
  val_losses = np.loadtxt(os.path.join(results_dir, va_files[-1]))
  te_losses = np.loadtxt(os.path.join(results_dir, te_files[-1]))

trained_epochs = int(np.floor(tr_batches.shape[0] / batches_per_epoch))
trained_batches = trained_epochs * batches_per_epoch
whole_epochs = np.split(tr_batches[:trained_batches], trained_epochs)
leftover = tr_batches[trained_batches:]

tr_losses = np.mean(np.array(whole_epochs), axis=1)

tr_val_epochs = np.arange(trained_epochs) + 1

h1 = plt.figure(figsize=(14, 8))
ax1 = h1.add_subplot(111)
plt.plot(tr_val_epochs, tr_losses, tr_val_epochs, val_losses, '--', linewidth=2)
plt.legend(['loss', 'val_loss'], loc='best')

plt.suptitle(
    'Training and validation losses. Test loss = {}'.format(np.mean(te_losses)))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss values')

ax1.set_xlim(tr_val_epochs[0], tr_val_epochs[-1])
major_xticks = np.arange(tr_val_epochs[0], tr_val_epochs[-1] + 1, 1)
ax1.set_xticks(major_xticks)
ax1.tick_params(which='both', direction='out')

ax1.grid(which='both', ls='-')

# TODO Save model parameters in TF training
model_description = 'tf_seq2seq'
params_loss = 'mse'
optimizer_name = 'adam'
learning_rate = 0.001

fig_filename = os.path.join(results_dir,
                            model_description + '_' + params_loss + '_' +
                            optimizer_name + '_epochs_' + str(epoch) + '_lr_' +
                            str(learning_rate) + '_graph')

plt.savefig(fig_filename + '.eps', bbox_inches='tight')
# plt.savefig(fig_filename + '.png', bbox_inches='tight')

# plt.show()
plt.close(h1)
