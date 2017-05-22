import gzip
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TKagg')


def show_attention():
  # Load attentions
  print('Loading attentions to pickle file')
  with gzip.open(
      os.path.join('training_results', 'torch_train', 'attentions.pkl.gz'),
      'r') as att_file:
    attentions = pickle.load(att_file)

  # Set up figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(np.mean(np.array(attentions),axis=(0,1)), cmap='bone')
  fig.colorbar(cax)

  # # Set up axes
  # ax.set_xticklabels([''] + input_sentence.split(' ') +
  #                    ['<EOS>'], rotation=90)
  # ax.set_yticklabels([''] + output_words)
  #
  # # Show label at every tick
  # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


show_attention()
