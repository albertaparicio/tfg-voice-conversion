#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

epoch = np.loadtxt('epoch.csv', delimiter=',')
loss = np.loadtxt('loss.csv', delimiter=',')
val_loss = np.loadtxt('val_loss.csv', delimiter=',')

assert (val_loss.size == loss.size)

plt.plot(epoch, loss, epoch, val_loss, '--', linewidth=2)
plt.legend(['Training loss', 'Validation loss'])
plt.grid(b='on')
plt.savefig('losses.png', bbox_inches='tight')
plt.show()

exit()
