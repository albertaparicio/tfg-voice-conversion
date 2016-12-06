import matplotlib.pyplot as plt
import numpy as np

epochs = 50
learning_rate = 0.002
optimizer = 'adamax'
loss = 'mse'

epoch = np.loadtxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
                   str(epochs) + '_lr_' + str(learning_rate) +
                   '_epochs.csv', delimiter=',')
losses = np.loadtxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
                    str(epochs) + '_lr_' + str(learning_rate) +
                    '_loss.csv', delimiter=',')
val_losses = np.loadtxt('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
                        str(epochs) + '_lr_' + str(learning_rate) +
                        '_val_loss.csv', delimiter=',')

assert (val_losses.size == losses.size)

plt.plot(epoch, losses, epoch, val_losses, '--', linewidth=2)
plt.legend(['Training loss', 'Validation loss'])
plt.grid(b='on')
plt.title('Loss: ' + loss + ', Optimizer: ' + optimizer + ', Epochs: ' + str(epochs) + ', Learning rate: ' + str(
    learning_rate))
plt.savefig('training_results/seq2seq_' + loss + '_' + optimizer + '_epochs_' +
            str(epochs) + '_lr_' + str(learning_rate) +
            '_graph.png', bbox_inches='tight')
plt.show()

exit()
