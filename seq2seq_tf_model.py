# Created by albert aparicio on 31/03/17
# coding: utf-8

# This script defines an Sequence-to-Sequence model, using the implemented
# encoders and decoders from TensorFlow

# TODO Document and explain steps

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import logging

import tensorflow as tf

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class Seq2Seq(object):
  # TODO Figure out suitable values for attention length and size
  def __init__(self, enc_rnn_layers, dec_rnn_layers, rnn_size, seq_length,
               params_length, cell_type='lstm', batch_size=20,
               learning_rate=0.001, dropout=0.5, optimizer='adam', clip_norm=5,
               attn_length=500, attn_size=100, infer=False):
    """
    infer: only True if used for test or predictions. False to train.
    """
    self.rnn_size = rnn_size
    self.attn_length = attn_length
    self.attn_size = attn_size
    # Number of layers in encoder and decoder
    self.enc_rnn_layers = enc_rnn_layers
    self.dec_rnn_layers = dec_rnn_layers
    self.infer = infer
    if infer:
      self.keep_prob = tf.Variable(1., trainable=False)
    else:
      self.keep_prob = tf.Variable((1. - dropout), trainable=False)

    self.dropout = dropout
    self.cell_type = cell_type
    self.batch_size = batch_size
    self.clip_norm = clip_norm
    self.learning_rate = learning_rate
    self.seq_length = seq_length
    self.parameters_length = params_length
    self.gtruth = tf.placeholder(tf.float32,
                                 [self.batch_size,
                                  self.seq_length,
                                  self.parameters_length])
    self.encoder_inputs = tf.placeholder(tf.float32,
                                         [batch_size,
                                          self.seq_length,
                                          self.parameters_length])
    self.decoder_inputs = tf.placeholder(tf.float32,
                                         [batch_size,
                                          self.seq_length,
                                          self.parameters_length])
    # attention_states: 3D Tensor [batch_size x attn_length x attn_size]
    self.attention_states = tf.placeholder(tf.float32,
                                           [batch_size,
                                            self.attn_length,
                                            self.attn_size])
    self.prediction = self.inference()
    self.loss = self.mse_loss(self.gtruth, self.prediction)

    tvars = tf.trainable_variables()
    grads = []
    for grad in tf.gradients(self.loss, tvars):
      if grad is not None:
        grads.append(tf.clip_by_norm(grad, self.clip_norm))
      else:
        grads.append(grad)

    self.optimizer = optimizer
    # set up a variable to make the learning rate evolve during training
    self.curr_lr = tf.Variable(self.learning_rate, trainable=False)
    self.opt = tf.train.AdamOptimizer(self.curr_lr)

    self.train_op = self.opt.apply_gradients(zip(grads, tvars))
    self.enc_state = None  # To be assigned a value in self.inference()

  @staticmethod
  def build_multirnn_block(rnn_size, rnn_layers, cell_type):
    if cell_type == 'gru':
      cell = tf.contrib.rnn.GRUCell(rnn_size)
    elif cell_type == 'lstm':
      cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    else:
      raise ValueError("The selected cell type '%s' is not supported"
                       % cell_type)

    if rnn_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layers,
                                         state_is_tuple=True)
    return cell

  @staticmethod
  def mse_loss(gtruth, prediction):
    return tf.reduce_mean(tf.squared_difference(gtruth, prediction))

  def inference(self):
    from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import \
      attention_decoder as tf_attention_decoder
    from tensorflow.contrib.rnn.python.ops.core_rnn import \
      static_rnn as tf_static_rnn

    with tf.variable_scope("encoder"):
      enc_cell = Seq2Seq.build_multirnn_block(self.rnn_size,
                                              self.enc_rnn_layers,
                                              self.cell_type)
      self.enc_zero = enc_cell.zero_state(self.batch_size, tf.float32)

      enc_out, enc_state = tf_static_rnn(enc_cell,
                                         self.encoder_inputs,
                                         initial_state=self.enc_zero,
                                         sequence_length=self.seq_length)
    # this op is created to visualize the thought vectors
    self.enc_state = enc_state
    logging.info('enc out tensor shape: ' + enc_out.get_shape())
    # print('enc out tensor shape: ', enc_out.get_shape())

    dec_cell = Seq2Seq.build_multirnn_block(self.rnn_size,
                                            self.dec_rnn_layers,
                                            self.cell_type)
    if self.dropout > 0:
      # print('Applying dropout {} to decoder'.format(self.dropout))
      logging.info('Applying dropout {} to decoder'.format(self.dropout))
      dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                               input_keep_prob=self.keep_prob)

    dec_cell = tf.contrib.rnn.OutputProjectionWrapper(dec_cell,
                                                      self.parameters_length)
    if not self.infer:
      def loop_function(prev, _):
        return prev
    else:
      loop_function = None

    dec_out, dec_state = tf_attention_decoder(
      self.decoder_inputs, enc_state, cell=dec_cell,
      attention_states=self.attention_states, loop_function=loop_function
    )

    # print('dec_state shape: ', dec_state[0].get_shape())
    # merge outputs into a tensor and transpose to be [B, seq_length, out_dim]
    dec_outputs = tf.transpose(tf.stack(dec_out), [1, 0, 2])
    # print('dec outputs shape: ', dec_outputs.get_shape())
    logging.info('dec outputs shape: ', dec_outputs.get_shape())

    return dec_outputs
