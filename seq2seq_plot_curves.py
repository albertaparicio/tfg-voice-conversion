# Created by Albert Aparicio on 6/12/16
# coding: utf-8

# This script takes the results of a training and plots its loss curves

import h5py
import matplotlib.pyplot as plt
import numpy as np

model_description = 'seq2seq_pretrain'

with h5py.File('training_results/' + model_description + '_training_params.h5',
               'r') as f:
    params_loss = f.attrs.get('params_loss').decode('utf-8')
    flags_loss = f.attrs.get('flags_loss').decode('utf-8')
    optimizer_name = f.attrs.get('optimizer').decode('utf-8')
    nb_epochs = f.attrs.get('epochs')
    learning_rate = f.attrs.get('learning_rate')
    metrics_names = [name.decode('utf-8') for name in
                     f.attrs.get('metrics_names')]

    f.close()

epoch = np.loadtxt('training_results/' + model_description + '_' + params_loss
                   + '_' + flags_loss + '_' + optimizer_name + '_epochs_' +
                   str(nb_epochs) + '_lr_' + str(learning_rate) + '_epochs.csv',
                   delimiter=',')
losses = np.loadtxt('training_results/' + model_description + '_' + params_loss
                    + '_' + flags_loss + '_' + optimizer_name + '_epochs_' +
                    str(nb_epochs) + '_lr_' + str(learning_rate) +
                    '_loss.csv', delimiter=',')
val_losses = np.loadtxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_val_loss.csv', delimiter=',')
mcd = np.loadtxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_mcd.csv', delimiter=',')
rmse = np.loadtxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_rmse.csv', delimiter=',')
acc = np.loadtxt(
    'training_results/' + model_description + '_' + params_loss + '_' +
    flags_loss + '_' + optimizer_name + '_epochs_' + str(nb_epochs) + '_lr_' +
    str(learning_rate) + '_acc.csv', delimiter=',')

assert (val_losses.size == losses.size)

# ##############################################
# # TODO Comment after dev
# metrics_names = ['loss', 'params_output_loss', 'flags_output_loss']
#
# ##############################################

# Losses plot
h1 = plt.figure(figsize=(14, 8))
ax1 = h1.add_subplot(111)

plt.plot(epoch, losses, epoch, val_losses, '--', linewidth=2)

# Prepare legend
legend_list = list(metrics_names)  # We use list() to make a copy

for name in metrics_names:
    legend_list.append('val_' + name)

plt.legend(legend_list, loc='best')

plt.suptitle('Parameters loss: ' + params_loss + ', Flags loss: ' + flags_loss +
             ', Optimizer: ' + optimizer_name + ', Epochs: ' + str(nb_epochs) +
             ', Learning rate: ' + str(learning_rate))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss values')

ax1.set_xlim(0, 19)
major_xticks = np.arange(0, 20, 1)
ax1.set_xticks(major_xticks)
ax1.tick_params(which='both', direction ='out')

ax1.grid(which='both', ls='-')

plt.savefig('training_results/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epochs_' +
            str(nb_epochs) + '_lr_' + str(learning_rate) + '_graph.eps',
            bbox_inches='tight')
# plt.show()
plt.close(h1)

# Metrics plot
h2 = plt.figure(figsize=(10, 5))
ax2 = h2.add_subplot(111)

plt.plot(epoch, mcd)  # , epoch, rmse, epoch, acc)
plt.legend(['MCD (dB)'], loc='best')
# , 'RMSE', 'Accuracy'
plt.suptitle("Cepstral features' MCD", fontsize = 12)
# , RMSE and ACC
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MCD (dB)')

ax2.set_xlim(0, 19)
major_xticks = np.arange(0, 20, 1)

major_yticks = np.arange(np.floor(np.min(mcd)), np.ceil(np.max(mcd)), 0.2)

ax2.set_xticks(major_xticks)
ax2.set_yticks(major_yticks)

ax2.tick_params(which='both', direction ='out')

ax2.grid(which='both', ls='-')
plt.savefig('training_results/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epochs_' +
            str(nb_epochs) + '_lr_' + str(learning_rate) + '_mcd.eps',
            bbox_inches='tight')

plt.close(h2)

h2 = plt.figure(figsize=(10, 5))
ax2 = h2.add_subplot(111)

plt.plot(epoch, rmse)
plt.legend(['RMSE'], loc='best')
# , 'RMSE', 'Accuracy'
plt.suptitle("Pitch Root Mean Square Error (RMSE)", fontsize=12)
# , RMSE and ACC
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Root Mean Square Error (RMSE)')

ax2.set_xlim(0, 19)

major_xticks = np.arange(0, 20, 1)

major_yticks = np.arange(0, np.ceil(np.max(rmse*100))/100, 0.01)

ax2.set_xticks(major_xticks)
ax2.set_yticks(major_yticks)

ax2.tick_params(which='both', direction ='out')

ax2.grid(which='both', ls='-')
plt.savefig('training_results/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epochs_' +
            str(nb_epochs) + '_lr_' + str(learning_rate) + '_rmse.eps',
            bbox_inches='tight')
plt.close(h2)

h2 = plt.figure(figsize=(10, 5))
ax2 = h2.add_subplot(111)

plt.plot(epoch, acc)
plt.legend(['Accuracy'], loc='best')
plt.suptitle("U/V Flag Accuracy", fontsize=12)
# , RMSE and ACC
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

ax2.set_xlim(0, 19)

major_xticks = np.arange(0, 20, 1)

major_yticks = np.arange(
    np.floor(np.min(acc*100))/100,
    1.005,
    0.005
)

ax2.set_xticks(major_xticks)
ax2.set_yticks(major_yticks)

ax2.tick_params(which='both', direction ='out')

ax2.grid(which='both', ls='-')
plt.savefig('training_results/' + model_description + '_' + params_loss + '_' +
            flags_loss + '_' + optimizer_name + '_epochs_' +
            str(nb_epochs) + '_lr_' + str(learning_rate) + '_acc.eps',
            bbox_inches='tight')
plt.close(h2)
exit()
