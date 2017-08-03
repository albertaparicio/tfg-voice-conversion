# -*- coding: utf-8 -*-
# TODO Add argparser
"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

This is made possible by the simple but powerful idea of the `sequence
to sequence network <http://arxiv.org/abs/1409.3215>`__, in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a vector,
and a decoder network unfolds that vector into a new sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt: 

To improve upon this model we'll use an `attention
mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
learn to focus over a specific range of the input sequence.

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

-  http://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in 
general
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user


It would also be useful to know about Sequence to Sequence networks and
how they work:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__

You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.

And for more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__


**Requirements**
"""
from __future__ import division, print_function, unicode_literals

import argparse
import glob
import gzip
import os
import random
from sys import version_info

import h5py
import numpy as np
import torch
import torch.nn as nn
from ahoproc_tools import error_metrics
from tfglib.seq2seq_normalize import mask_data
from tfglib.utils import init_logger
from torch import optim
from torch.autograd import Variable

from seq2seq_dataloader import DataLoader
from seq2seq_pytorch_model import AttnDecoderRNN, EncoderRNN

use_cuda = torch.cuda.is_available()

# Conditional imports
if version_info.major > 2:
  import pickle
else:
  import cPickle as pickle

logger, opts = None, None

if __name__ == '__main__':
  # logger.debug('Before parsing args')
  parser = argparse.ArgumentParser(
      description="Convert voice signal with seq2seq model")
  parser.add_argument('--train_data_path', type=str,
                      default="tcstar_data_trim/training/")
  parser.add_argument('--train_out_file', type=str,
                      default="tcstar_data_trim/seq2seq_train_datatable")
  parser.add_argument('--test_data_path', type=str,
                      default="tcstar_data_trim/test/")
  parser.add_argument('--test_out_file', type=str,
                      default="tcstar_data_trim/seq2seq_test_datatable")
  parser.add_argument('--val_fraction', type=float, default=0.25)
  parser.add_argument('--save-h5', dest='save_h5', action='store_true',
                      help='Save dataset to .h5 file')
  parser.add_argument('--max_seq_length', type=int, default=500)
  parser.add_argument('--params_len', type=int, default=44)
  # parser.add_argument('--patience', type=int, default=4,
  #                     help="Patience epochs to do validation, if validation "
  #                          "score is worse than train for patience epochs "
  #                          ", quit training. (Def: 4).")
  # parser.add_argument('--enc_rnn_layers', type=int, default=1)
  # parser.add_argument('--dec_rnn_layers', type=int, default=1)
  parser.add_argument('--hidden_size', type=int, default=256)
  # parser.add_argument('--cell_type', type=str, default="lstm")
  parser.add_argument('--batch_size', type=int, default=10)
  parser.add_argument('--epoch', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=0.0005)
  # parser.add_argument('--dropout', type=float, default=0)
  parser.add_argument('--teacher_forcing_ratio', type=float, default=1)
  parser.add_argument('--SOS_token', type=int, default=0)

  # parser.add_argument('--optimizer', type=str, default="adam")
  # parser.add_argument('--clip_norm', type=float, default=5)
  # parser.add_argument('--attn_length', type=int, default=500)
  # parser.add_argument('--attn_size', type=int, default=256)
  # parser.add_argument('--save_every', type=int, default=100)
  parser.add_argument('--no-train', dest='do_train',
                      action='store_false', help='Flag to train or not.')
  parser.add_argument('--no-test', dest='do_test',
                      action='store_false', help='Flag to test or not.')
  parser.add_argument('--save_path', type=str, default="training_results")
  parser.add_argument('--pred_path', type=str, default="torch_predicted")
  # parser.add_argument('--tb_path', type=str, default="")
  parser.add_argument('--log', type=str, default="INFO")
  parser.add_argument('--load_model', dest='load_model', action='store_true',
                      help='Load previous model before training')
  parser.add_argument('--server', dest='server', action='store_true',
                      help='Commands to be run or not run if we are running '
                           'on server')

  parser.set_defaults(do_train=True, do_test=True, save_h5=False,
                      server=False)  # ,
  # load_model=False)
  opts = parser.parse_args()

  # Initialize logger
  logger_level = opts.log
  logger = init_logger(name=__name__, level=opts.log)

  logger.debug('Parsed arguments')
  if not os.path.exists(os.path.join(opts.save_path, 'torch_train')):
    os.makedirs(os.path.join(opts.save_path, 'torch_train'))
  # save config
  with gzip.open(os.path.join(opts.save_path, 'torch_train', 'config.pkl.gz'),
                 'wb') as cf:
    pickle.dump(opts, cf)


def main(args):
  logger.debug('Main')

  # If-else for training and testing
  if args.do_train:
    dl = DataLoader(args, logger_level=args.log,
                    max_seq_length=args.max_seq_length)

    encoder1 = EncoderRNN(args.params_len, args.hidden_size, args.batch_size)
    attn_decoder1 = AttnDecoderRNN(args.hidden_size, args.params_len,
                                   batch_size=args.batch_size, n_layers=1,
                                   max_length=args.max_seq_length,
                                   dropout_p=0.1)

    if use_cuda:
      encoder1 = encoder1.cuda()
      attn_decoder1 = attn_decoder1.cuda()

    trained_encoder, trained_decoder = train_epochs(dl, encoder1, attn_decoder1)

  if args.do_test:
    # TODO What do we do for testing?
    # pass
    dl = DataLoader(args, logger_level=args.log, test=True,
                    max_seq_length=args.max_seq_length)

    if args.load_model:
      encoder = EncoderRNN(args.params_len, args.hidden_size, args.batch_size)
      decoder = AttnDecoderRNN(args.hidden_size, args.params_len,
                               batch_size=args.batch_size, n_layers=1,
                               max_length=args.max_seq_length,
                               dropout_p=0.1)
      if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    else:
      encoder = trained_encoder
      decoder = trained_decoder

    test(encoder, decoder, dl)


######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <http://opendata.stackexchange.com/questions/3888/dataset-of
# -sentences-translated-into-many-languages>`__
# pointed me to the open translation site http://tatoeba.org/ which has
# downloads available at http://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: http://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    Je suis froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

# EOS_token = 1
#
#
# class Lang:
#   def __init__(self, name):
#     self.name = name
#     self.word2index = {}
#     self.word2count = {}
#     self.index2word = {0: "SOS", 1: "EOS"}
#     self.n_words = 2  # Count SOS and EOS
#
#   def add_sentence(self, sentence):
#     for word in sentence.split(' '):
#       self.add_word(word)
#
#   def add_word(self, word):
#     if word not in self.word2index:
#       self.word2index[word] = self.n_words
#       self.word2count[word] = 1
#       self.index2word[self.n_words] = word
#       self.n_words += 1
#     else:
#       self.word2count[word] += 1
#
#
# ######################################################################
# # The files are all in Unicode, to simplify we will turn Unicode
# # characters to ASCII, make everything lowercase, and trim most
# # punctuation.
# #
#
# # Turn a Unicode string to plain ASCII, thanks to
# # http://stackoverflow.com/a/518232/2809427
# def unicode_to_ascii(s):
#   return ''.join(
#       c for c in unicodedata.normalize('NFD', s)
#       if unicodedata.category(c) != 'Mn'
#       )
#
#
# # Lowercase, trim, and remove non-letter characters
# def normalize_string(s):
#   s = unicode_to_ascii(s.lower().strip())
#   s = re.sub(r"([.!?])", r" \1", s)
#   s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#   return s
#
#
# ######################################################################
# # To read the data file we will split the file into lines, and then split
# # lines into pairs. The files are all English → Other Language, so if we
# # want to translate from Other Language → English I added the ``reverse``
# # flag to reverse the pairs.
# #
#
# def read_langs(lang1, lang2, reverse=False):
#   print("Reading lines...")
#
#   # Read the file and split into lines
#   lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
#     read().strip().split('\n')
#
#   # Split every line into pairs and normalize
#   pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
#
#   # Reverse pairs, make Lang instances
#   if reverse:
#     pairs = [list(reversed(p)) for p in pairs]
#     input_lang = Lang(lang2)
#     output_lang = Lang(lang1)
#   else:
#     input_lang = Lang(lang1)
#     output_lang = Lang(lang2)
#
#   return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#
#
# eng_prefixes = (
#   "i am ", "i m ",
#   "he is", "he s ",
#   "she is", "she s",
#   "you are", "you re ",
#   "we are", "we re ",
#   "they are", "they re "
#   )
#
#
# def filter_pair(p):
#   return len(p[0].split(' ')) < opts.max_seq_length and \
#          len(p[1].split(' ')) < opts.max_seq_length and \
#          p[1].startswith(eng_prefixes)
#
#
# def filter_pairs(pairs):
#   return [pair for pair in pairs if filter_pair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#
#
# def prepare_data(lang1, lang2, reverse=False):
#   input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
#   print("Read %s sentence pairs" % len(pairs))
#   pairs = filter_pairs(pairs)
#   print("Trimmed to %s sentence pairs" % len(pairs))
#   print("Counting words...")
#   for pair in pairs:
#     input_lang.add_sentence(pair[0])
#     output_lang.add_sentence(pair[1])
#   print("Counted words:")
#   print(input_lang.name, input_lang.n_words)
#   print(output_lang.name, output_lang.n_words)
#   return input_lang, output_lang, pairs
#
#
# input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
# print(random.choice(pairs))


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#
#
# def indexes_from_sentence(lang, sentence):
#   return [lang.word2index[word] for word in sentence.split(' ')]
#
#
# def variable_from_sentence(lang, sentence):
#   indexes = indexes_from_sentence(lang, sentence)
#   indexes.append(EOS_token)
#   result = Variable(torch.LongTensor(indexes).view(-1, 1))
#   if use_cuda:
#     return result.cuda()
#   else:
#     return result
#
#
# def variables_from_pair(pair):
#   input_variable = variable_from_sentence(input_lang, pair[0])
#   target_variable = variable_from_sentence(output_lang, pair[1])
#   return input_variable, target_variable


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# decoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads
# /papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#



def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer,
          decoder_optimizer, criterion, max_length):
  encoder_hidden = encoder.init_hidden()

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
        input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  # decoder_input = Variable(torch.LongTensor([[opts.SOS_token]]))
  decoder_input = Variable(torch.zeros(opts.batch_size, opts.params_len))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random() < opts.teacher_forcing_ratio \
    else False

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_output, encoder_outputs)
      loss += criterion(decoder_output, target_variable[di])
      decoder_input = target_variable[di]  # Teacher forcing

  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_output, encoder_outputs)
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]

      decoder_input = Variable(torch.LongTensor([[ni]]))
      decoder_input = decoder_input.cuda() if use_cuda else decoder_input

      loss += criterion(decoder_output[0], target_variable[di])
      # if ni == EOS_token:
      #   break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def as_minutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def time_since(since, percent):
  now = time.time()
  s = now - since
  es = s / percent
  rs = es - s
  return '%s (ETA: %s)' % (as_minutes(s), as_minutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of epochs, time so far, estimated time) and average loss.
#

def train_epochs(dataloader, encoder, decoder):
  start = time.time()
  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every
  batch_idx = 0
  total_batch_idx = 0
  curr_epoch = 0
  b_epoch = dataloader.train_batches_per_epoch

  # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=opts.learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=opts.learning_rate)
  # training_pairs = [variables_from_pair(random.choice(pairs))
  #                   for _ in range(n_epochs)]
  # criterion = nn.NLLLoss()
  criterion = nn.MSELoss()

  # Train on dataset batches
  for src_batch, src_batch_seq_len, trg_batch, trg_mask in \
      dataloader.next_batch():
    if curr_epoch == 0 and batch_idx == 0:
      logger.info(
          'Batches per epoch: {}'.format(b_epoch))
      logger.info(
          'Total batches: {}'.format(b_epoch * opts.epoch))
    # beg_t = timeit.default_timer()

    # for epoch in range(1, n_epochs + 1):
    # training_pair = training_pairs[epoch - 1]
    # input_variable = training_pair[0]
    # target_variable = training_pair[1]

    # Transpose data to be shaped (max_seq_length, num_sequences, params_len)
    input_variable = Variable(
        torch.from_numpy(src_batch[:, :, 0:44]).float()
        ).transpose(1, 0).contiguous()
    target_variable = Variable(
        torch.from_numpy(trg_batch).float()
        ).transpose(1, 0).contiguous()

    input_variable = input_variable.cuda() if use_cuda else input_variable
    target_variable = target_variable.cuda() if use_cuda else target_variable

    loss = train(input_variable, target_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion,
                 opts.max_seq_length)
    print_loss_total += loss
    # plot_loss_total += loss

    print_loss_avg = print_loss_total / (total_batch_idx + 1)
    plot_losses.append(print_loss_avg)

    logger.info(
        'Batch {:2.0f}/{:2.0f} - Epoch {:2.0f}/{:2.0f} ({:3.2%}) - Loss={'
        ':.8f} - Time: {'
        '!s}'.format(
            batch_idx,
            b_epoch,
            curr_epoch + 1,
            opts.epoch,
            ((batch_idx % b_epoch) + 1) / b_epoch,
            print_loss_avg,
            time_since(start,
                       (total_batch_idx + 1) / (b_epoch * opts.epoch))))

    if batch_idx >= b_epoch:
      curr_epoch += 1
      batch_idx = 0
      print_loss_total = 0

      # Save model
      # Instructions for saving and loading a model:
      # http://pytorch.org/docs/notes/serialization.html
      # with gzip.open(
      enc_file = os.path.join(opts.save_path, 'torch_train',
                              'encoder_{}.pkl'.format(
                                curr_epoch))  # , 'wb') as enc:
      torch.save(encoder.state_dict(), enc_file)
      # with gzip.open(
      dec_file = os.path.join(opts.save_path, 'torch_train',
                              'decoder_{}.pkl'.format(
                                curr_epoch))  # , 'wb') as dec:
      torch.save(decoder.state_dict(), dec_file)

      # TODO Validation?

    batch_idx += 1
    total_batch_idx += 1

    if curr_epoch >= opts.epoch:
      logger.info('Finished epochs -> BREAK')
      break

  if not opts.server:
    show_plot(plot_losses)

  else:
    save_path = os.path.join(opts.save_path, 'torch_train', 'graphs')

    if not os.path.exists(save_path):
      os.makedirs(save_path)

    np.savetxt(os.path.join(save_path, 'train_losses' + '.csv'), plot_losses)

  return encoder, decoder


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

if not opts.server:
  import matplotlib

  matplotlib.use('TKagg')
  import matplotlib.pyplot as plt
  import matplotlib.ticker as ticker


  def show_plot(points, filename='train_loss'):
    if not os.path.exists(
        os.path.join(opts.save_path, 'torch_train', 'graphs')):
      os.makedirs(os.path.join(opts.save_path, 'torch_train', 'graphs'))

    plt.figure()
    # fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.grid(b=True)
    plt.savefig(
        os.path.join(opts.save_path, 'torch_train', 'graphs',
                     filename + '.eps'),
        bbox_inches='tight')


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def test(encoder, decoder, dl):
  if opts.load_model:
    # Get filenames of last epoch files
    enc_file = sorted(glob.glob(
        os.path.join(opts.save_path, 'torch_train', 'encoder*.pkl')))[-1]
    dec_file = sorted(glob.glob(
        os.path.join(opts.save_path, 'torch_train', 'decoder*.pkl')))[-1]

    # Open model files and load
    # with gzip.open(enc_file, 'r') as enc:
    #   enc_f = pickle.load(enc)
    encoder.load_state_dict(torch.load(enc_file))
    #
    # with gzip.open(dec_file, 'wb') as dec:
    decoder.load_state_dict(torch.load(dec_file))

  batch_idx = 0
  n_batch = 0
  attentions = []

  for (src_batch_padded, src_batch_seq_len, trg_batch, trg_mask) in dl.next_batch(
      test=True):
    src_batch = []

    # Take the last `seq_len` timesteps of each sequence to remove padding
    for i in range(src_batch_padded.shape[0]):
      src_batch.append(src_batch_padded[i,-src_batch_seq_len[i]:,:])

    # TODO Get filename from datatable
    f_name = format(n_batch + 1,
                    '0' + str(max(5, len(str(dl.src_test_data.shape[0])))))

    input_variable = Variable(
        torch.from_numpy(src_batch[:, :, 0:44]).float()
        ).transpose(1, 0).contiguous()
    input_variable = input_variable.cuda() if use_cuda else input_variable

    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(
        torch.zeros(opts.max_seq_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
      encoder_output, encoder_hidden = encoder(input_variable[ei],
                                               encoder_hidden)
      encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    # decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = Variable(torch.zeros(opts.batch_size, opts.params_len))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_frames = []
    decoder_attentions = torch.zeros(opts.max_seq_length, opts.batch_size,
                                     opts.max_seq_length)

    for di in range(opts.max_seq_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_output, encoder_outputs)
      decoder_attentions[di] = decoder_attention.data
      # topv, topi = decoder_output.data.topk(1)
      # ni = topi[0][0]
      # if ni == EOS_token:
      #   decoded_frames.append('<EOS>')
      #   break
      # else:
      decoded_frames.append(decoder_output.data.cpu().numpy())

      # decoder_input = Variable(torch.LongTensor([[ni]]))
      decoder_input = decoder_output
      decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Decode output frames
    predictions = np.array(decoded_frames).transpose((1, 0, 2))
    attentions.append(decoder_attentions[:di + 1].numpy().transpose((1, 0, 2)))

    # TODO Decode speech data and display attentions
    # Save original U/V flags to save them to file
    raw_uv_flags = predictions[:, :, 42]

    # Unscale target and predicted parameters
    for i in range(predictions.shape[0]):

      src_spk_index = int(src_batch[i, 0, 44])
      trg_spk_index = int(src_batch[i, 0, 45])

      # Prepare filename
      # Get speakers names
      src_spk_name = dl.s2s_datatable.src_speakers[src_spk_index]
      trg_spk_name = dl.s2s_datatable.trg_speakers[trg_spk_index]

      # Make sure the save directory exists
      tf_pred_path = os.path.join(opts.test_data_path, opts.pred_path)

      if not os.path.exists(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name)):
        os.makedirs(
            os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name))

        # # TODO Get filename from datatable
        # f_name = format(i + 1, '0' + str(
        # max(5, len(str(dl.src_test_data.shape[0])))))

      with h5py.File(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name,
                       f_name + '_' + str(i) + '.h5'), 'w') as file:
        file.create_dataset('predictions', data=predictions[i],
                            compression="gzip",
                            compression_opts=9)
        file.create_dataset('target', data=trg_batch[i],
                            compression="gzip",
                            compression_opts=9)
        file.create_dataset('mask', data=trg_mask[i],
                            compression="gzip",
                            compression_opts=9)

        trg_spk_max = dl.train_trg_speakers_max[trg_spk_index, :]
        trg_spk_min = dl.train_trg_speakers_min[trg_spk_index, :]

        trg_batch[i, :, 0:42] = (trg_batch[i, :, 0:42] * (
          trg_spk_max - trg_spk_min)) + trg_spk_min

        predictions[i, :, 0:42] = (predictions[i, :, 0:42] * (
          trg_spk_min - trg_spk_min)) + trg_spk_min

        # Round U/V flags
        predictions[i, :, 42] = np.round(predictions[i, :, 42])

        # Remove padding in prediction and target parameters
        masked_trg = mask_data(trg_batch[i], trg_mask[i])
        trg_batch[i] = np.ma.filled(masked_trg, fill_value=0.0)
        unmasked_trg = np.ma.compress_rows(masked_trg)

        masked_pred = mask_data(predictions[i], trg_mask[i])
        predictions[i] = np.ma.filled(masked_pred, fill_value=0.0)
        unmasked_prd = np.ma.compress_rows(masked_pred)

        # # Apply U/V flag to lf0 and mvf params
        # unmasked_prd[:, 40][unmasked_prd[:, 42] == 0] = -1e10
        # unmasked_prd[:, 41][unmasked_prd[:, 42] == 0] = 1000

        # Apply ground truth flags to prediction
        unmasked_prd[:, 40][unmasked_trg[:, 42] == 0] = -1e10
        unmasked_prd[:, 41][unmasked_trg[:, 42] == 0] = 1000

        file.create_dataset('unmasked_prd', data=unmasked_prd,
                            compression="gzip",
                            compression_opts=9)
        file.create_dataset('unmasked_trg', data=unmasked_trg,
                            compression="gzip",
                            compression_opts=9)
        file.create_dataset('trg_max', data=trg_spk_max,
                            compression="gzip",
                            compression_opts=9)
        file.create_dataset('trg_min', data=trg_spk_min,
                            compression="gzip",
                            compression_opts=9)
        file.close()

      # Save predictions to files
      np.savetxt(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name,
                       f_name + '_' + str(i) + '.vf.dat'),
          unmasked_prd[:, 41]
          )
      np.savetxt(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name,
                       f_name + '_' + str(i) + '.lf0.dat'),
          unmasked_prd[:, 40]
          )
      np.savetxt(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name,
                       f_name + '_' + str(i) + '.mcp.dat'),
          unmasked_prd[:, 0:40],
          delimiter='\t'
          )
      np.savetxt(
          os.path.join(tf_pred_path, src_spk_name + '-' + trg_spk_name,
                       f_name + '_' + str(i) + '.uv.dat'),
          raw_uv_flags[i, :]
          )

    # Display metrics
    print('Num - {}'.format(n_batch))

    print('MCD = {} dB'.format(
        error_metrics.MCD(unmasked_trg[:, 0:40].reshape(-1, 40),
                          unmasked_prd[:, 0:40].reshape(-1, 40))))
    acc, _, _, _ = error_metrics.AFPR(unmasked_trg[:, 42].reshape(-1, 1),
                                      unmasked_prd[:, 42].reshape(-1, 1))
    print('U/V accuracy = {}'.format(acc))

    pitch_rmse = error_metrics.RMSE(
        np.exp(unmasked_trg[:, 40].reshape(-1, 1)),
        np.exp(unmasked_prd[:, 40].reshape(-1, 1)))
    print('Pitch RMSE = {}'.format(pitch_rmse))

    # Increase batch index
    if batch_idx >= dl.test_batches_per_epoch:
      break
    batch_idx += 1

    n_batch += 1

  # Dump attentions to pickle file
  logger.info('Saving attentions to pickle file')
  with gzip.open(
      os.path.join(opts.save_path, 'torch_train', 'attentions.pkl.gz'),
      'wb') as att_file:
    pickle.dump(attentions, att_file)


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

# def evaluate_randomly(encoder, decoder, n=10):
#   for i in range(n):
#     pair = random.choice(pairs)
#     print('>', pair[0])
#     print('=', pair[1])
#     output_words, attentions = evaluate(encoder, decoder, pair[0])
#     output_sentence = ' '.join(output_words)
#     print('<', output_sentence)
#     print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it's easier to run multiple experiments easier) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainEpochs`` again.
#



######################################################################
#

# evaluate_randomly(encoder1, attn_decoder1)

######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#
#
# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def show_attention():
  # Load attentions
  logger.info('Loading attentions to pickle file')
  with gzip.open(
      os.path.join(opts.save_path, 'torch_train', 'attentions.pkl.gz'),
      'r') as att_file:
    attentions = pickle.load(att_file)

  # Set up figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
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


# def evaluate_and_show_attention(input_sentence):
#   output_words, attentions = evaluate(
#       encoder1, attn_decoder1, input_sentence)
#   print('input =', input_sentence)
#   print('output =', ' '.join(output_words))
#   show_attention(input_sentence, output_words, attentions)
#
#
# evaluate_and_show_attention("elle a cinq ans de moins que moi .")
#
# evaluate_and_show_attention("elle est trop petit .")
#
# evaluate_and_show_attention("je ne crains pas de mourir .")
#
# evaluate_and_show_attention("c est un jeune directeur plein de talent .")

######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embedding pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#

if __name__ == '__main__':
  logger.debug('Before calling main')
  main(opts)
