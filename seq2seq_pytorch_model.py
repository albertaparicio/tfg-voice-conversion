# Created by albert aparicio on 31/03/17
# coding: utf-8
#
# Code from PyTorch's seq2seq tutorial
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size,batch_size, n_layers=1):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.batch_size = batch_size

    # self.embedding = nn.Embedding(input_size, hidden_size)
    # self.gru = nn.GRU(hidden_size, hidden_size)
    self.gru = nn.GRU(input_size, hidden_size,  n_layers)

  def forward(self, f_input, hidden):
    # embedded = self.embedding(f_input).view(1, 1, -1)
    # output = embedded
    output = f_input.view(1,f_input.size()[0],f_input.size()[1])

    # for i in range(self.n_layers):
    output, hidden = self.gru(output, hidden)
    return output, hidden

  def init_hidden(self):
    result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
    if use_cuda:
      return result.cuda()
    else:
      return result


######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size,batch_size, n_layers=1):
    super(DecoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.batch_size = batch_size

    # self.embedding = nn.Embedding(output_size, hidden_size)
    # self.gru = nn.GRU(hidden_size, hidden_size)
    self.gru = nn.GRU(output_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, f_input, hidden):
    # output = self.embedding(f_input).view(1, 1, -1)
    output = f_input.view(1,f_input.size()[0],f_input.size()[1])

    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
    # TODO Maybe the softmax must be removed
    output = self.softmax(self.out(output[0]))
    return output, hidden

  def init_hidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if use_cuda:
      return result.cuda()
    else:
      return result


######################################################################
# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, max_length,batch_size, n_layers=1,
               dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length
    self.batch_size = batch_size

    # self.embedding = nn.Embedding(self.output_size, self.hidden_size)

    self.attn = nn.Linear(self.hidden_size + self.output_size, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)

    # self.attn = nn.Linear(300, self.max_length)
    # self.attn_combine = nn.Linear(300, self.hidden_size)

    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, f_input, hidden, encoder_output, encoder_outputs):
    # embedded = self.embedding(f_input).view(1, 1, -1)
    # embedded = self.dropout(embedded)
    embedded = f_input.view(1,f_input.size()[0],f_input.size()[1])

    attn_weights = F.softmax(
        self.attn(torch.cat((embedded[0], hidden[0]), 1)))
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                             encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)

    # Project GRU output to our data's output size
    output = self.out(output[0])
    return output, hidden, attn_weights

  def init_hidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if use_cuda:
      return result.cuda()
    else:
      return result
