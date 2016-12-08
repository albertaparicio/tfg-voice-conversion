# Created by Albert Aparicio on 6/12/16
# coding: utf-8

# This script takes the results of a training and plots its loss curves

import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('training_results/seq2seq_training_params.h5', 'r') as f:
    epochs = f.attrs.get('epochs')
    learning_rate = f.attrs.get('learning_rate')
    optimizer = f.attrs.get('optimizer')
    loss = f.attrs.get('loss')

epoch = np.loadtxt('training_results/seq2seq_' + loss.decode('utf-8') +
                   '_' + optimizer.decode('utf-8') + '_epochs_' +
                   str(epochs) + '_lr_' + str(learning_rate) +
                   '_epochs.csv', delimiter=',')
losses = np.loadtxt('training_results/seq2seq_' + loss.decode('utf-8') +
                    '_' + optimizer.decode('utf-8') + '_epochs_' +
                    str(epochs) + '_lr_' + str(learning_rate) +
                    '_loss.csv', delimiter=',')
val_losses = np.loadtxt('training_results/seq2seq_' + loss.decode('utf-8') +
                        '_' + optimizer.decode('utf-8') + '_epochs_' +
                        str(epochs) + '_lr_' + str(learning_rate) +
                        '_val_loss.csv', delimiter=',')

assert (val_losses.size == losses.size)

plt.plot(epoch, losses, epoch, val_losses, '--', linewidth=2)
plt.legend(['Training loss', 'Validation loss'])
plt.grid(b='on')
plt.title('Loss: ' + loss.decode('utf-8') + ', Optimizer: ' +
          optimizer.decode('utf-8') + ', Epochs: ' + str(epochs) +
          ', Learning rate: ' + str(learning_rate)
          )
plt.savefig('training_results/seq2seq_' + loss.decode('utf-8') + '_' +
            optimizer.decode('utf-8') + '_epochs_' + str(epochs) + '_lr_' +
            str(learning_rate) + '_graph.png', bbox_inches='tight')
plt.show()

exit()
