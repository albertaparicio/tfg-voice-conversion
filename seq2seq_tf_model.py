# Created by albert aparicio on 31/03/17
# coding: utf-8

# This script defines an Sequence-to-Sequence model, using the implemented
# encoders and decoders from TensorFlow

# TODO Document and explain steps
# TODO Move this model to tfglib

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tfglib.seq2seq_datatable as s2s
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import \
  attention_decoder
from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn
from tfglib.seq2seq_normalize import maxmin_scaling
from tfglib.utils import init_logger


################################################################################
# Code from santi-pdp @ GitHub
# https://github.com/santi-pdp/word2phone/blob/master/model.py

def scalar_summary(name, x):
  try:
    summ = tf.summary.scalar(name, x)
  except AttributeError:
    summ = tf.scalar_summary(name, x)
  return summ


def histogram_summary(name, x):
  try:
    summ = tf.summary.histogram(name, x)
  except AttributeError:
    summ = tf.histogram_summary(name, x)
  return summ


################################################################################

class Seq2Seq(object):
  # TODO Figure out suitable values for attention length and size
  def __init__(self, enc_rnn_layers, dec_rnn_layers, rnn_size,
               seq_length, params_length, cell_type='lstm', batch_size=20,
               learning_rate=0.001, dropout=0.5, optimizer='adam', clip_norm=5,
               attn_length=500, attn_size=100, infer=False,
               logger_level='INFO'):
    """
    infer: only True if used for test or predictions. False to train.
    """
    self.logger = init_logger(name=__name__, level=logger_level)

    self.logger.debug('Seq2Seq init')
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

    self.seq_length = tf.placeholder(tf.int32, [self.batch_size])

    self.parameters_length = params_length

    self.gtruth = tf.placeholder(tf.float32,
                                 [self.batch_size,
                                  seq_length,
                                  self.parameters_length])

    # Ground truth summaries
    split_gtruth = tf.split(self.gtruth, self.parameters_length, axis=2,
                            name='gtruth_parameter')

    self.gtruth_summaries = []
    [self.gtruth_summaries.append(
        histogram_summary(split_tensor.name, split_tensor)) for split_tensor in
      split_gtruth]

    self.gtruth_masks = tf.placeholder(tf.float32,
                                       [self.batch_size, seq_length])

    self.encoder_inputs = [
      tf.placeholder(tf.float32, [batch_size, self.parameters_length]
                     ) for _ in range(seq_length)]

    # Encoder inputs summaries
    split_enc_inputs = tf.split(tf.stack(self.encoder_inputs, axis=1),
                                self.parameters_length, axis=2,
                                name='encoder_parameter')
    self.enc_inputs_summaries = []
    [self.enc_inputs_summaries.append(
        histogram_summary(split_tensor.name, split_tensor)) for split_tensor in
      split_enc_inputs]

    self.decoder_inputs = [
      tf.placeholder(tf.float32, [batch_size, self.parameters_length]
                     ) for _ in range(seq_length)]

    # To be assigned a value later
    self.enc_state_fw = None
    self.enc_state_bw = None
    self.encoder_vars = None
    self.enc_zero_fw = None
    self.enc_zero_bw = None
    self.encoder_state_summaries_fw = None
    self.encoder_state_summaries_bw = None
    self.decoder_outputs_summaries = []

    self.prediction = self.inference()

    self.loss = self.mse_loss(self.gtruth, self.gtruth_masks, self.prediction)
    self.val_loss = self.mse_loss(self.gtruth, self.gtruth_masks,
                                  self.prediction)

    self.loss_summary = scalar_summary('loss', self.loss)
    self.val_loss_summary = scalar_summary('val_loss', self.val_loss)

    tvars = tf.trainable_variables()
    grads = []
    for grad in tf.gradients(self.loss, tvars):
      # if grad is not None:
      #   grads.append(tf.clip_by_norm(grad, self.clip_norm))
      # else:
      grads.append(grad)

    self.optimizer = optimizer
    # set up a variable to make the learning rate evolve during training
    self.curr_lr = tf.Variable(self.learning_rate, trainable=False)
    self.opt = tf.train.AdamOptimizer(self.curr_lr)

    self.train_op = self.opt.apply_gradients(zip(grads, tvars))

  def build_multirnn_block(self, rnn_size, rnn_layers, cell_type,
                           activation=tf.tanh):
    self.logger.debug('Build RNN block')
    if cell_type == 'gru':
      cell = tf.contrib.rnn.GRUCell(rnn_size, activation=activation)
    elif cell_type == 'lstm':
      cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True,
                                          activation=activation)
    else:
      raise ValueError("The selected cell type '%s' is not supported"
                       % cell_type)

    if rnn_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layers,
                                         state_is_tuple=True)
    return cell

  def mse_loss(self, gtruth, gtruth_masks, prediction):
    """Mean squared error loss"""
    # Previous to the computation, the predictions are masked
    self.logger.debug('Compute loss')
    return tf.reduce_mean(tf.squared_difference(gtruth,
                                                prediction * tf.expand_dims(
                                                    gtruth_masks, -1)))

  def mae_loss(self, gtruth, gtruth_masks, prediction):
    """Mean absolute error loss"""
    # Previous to the computation, the predictions are masked
    self.logger.debug('Compute loss')
    return tf.reduce_mean(
        tf.abs((prediction * tf.expand_dims(gtruth_masks, -1)) - gtruth))

  def inference(self):
    self.logger.debug('Inference')
    self.logger.debug('Imported seq2seq model from TF')

    with tf.variable_scope("encoder"):
      enc_cell_fw = self.build_multirnn_block(self.rnn_size,
                                              self.enc_rnn_layers,
                                              self.cell_type)
      # enc_cell_bw = self.build_multirnn_block(self.rnn_size,
      #                                         self.enc_rnn_layers,
      #                                         self.cell_type)
      self.enc_zero_fw = enc_cell_fw.zero_state(self.batch_size, tf.float32)
      # self.enc_zero_bw = enc_cell_bw.zero_state(self.batch_size, tf.float32)

      self.logger.debug('Initialize encoder')

      # inputs = batch_norm(self.encoder_inputs, is_training=self.infer)
      # inputs = []
      # for tensor in self.encoder_inputs:
      #   inputs.append(batch_norm(tensor, is_training=self.infer))

      # enc_out_orig, enc_state_fw, enc_state_bw = static_bidirectional_rnn(
      #     cell_fw=enc_cell_fw, cell_bw=enc_cell_bw, inputs=inputs,
      #     initial_state_fw=self.enc_zero_fw,
      # initial_state_bw=self.enc_zero_bw,
      #     sequence_length=self.seq_length
      #     )
      enc_out, enc_state_fw = static_rnn(cell=enc_cell_fw,
                                         inputs=self.encoder_inputs,
                                         initial_state=self.enc_zero_fw,
                                         sequence_length=self.seq_length)

      # enc_out = []
      # for tensor in enc_out_orig:
      #   enc_out.append(batch_norm(tensor, is_training=self.infer))

    # This op is created to visualize the thought vectors
    self.enc_state_fw = enc_state_fw
    # self.enc_state_bw = enc_state_bw

    self.logger.info(
        'enc out (len {}) tensors shape: {}'.format(
            len(enc_out), enc_out[0].get_shape()
            ))
    # print('enc out tensor shape: ', enc_out.get_shape())

    self.encoder_state_summaries_fw = histogram_summary(
        'encoder_state_fw', enc_state_fw)
    # self.encoder_state_summaries_bw = histogram_summary(
    #     'encoder_state_bw', enc_state_bw)

    dec_cell = self.build_multirnn_block(self.rnn_size,
                                         self.dec_rnn_layers,
                                         self.cell_type)
    if self.dropout > 0:
      # print('Applying dropout {} to decoder'.format(self.dropout))
      self.logger.info('Applying dropout {} to decoder'.format(self.dropout))
      dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                               input_keep_prob=self.keep_prob)

    dec_cell = tf.contrib.rnn.OutputProjectionWrapper(dec_cell,
                                                      self.parameters_length)

    if self.infer:
      def loop_function(prev, _):
        return prev
    else:
      loop_function = None

    # First calculate a concatenation of encoder outputs to put attention on.
    # assert enc_cell_fw.output_size == enc_cell_bw.output_size
    # top_states = [
    #   tf.reshape(e, [-1, 1, enc_cell_fw.output_size +
    # enc_cell_bw.output_size])
    #   for e in enc_out
    #   ]
    top_states = [
      tf.reshape(e, [-1, 1, enc_cell_fw.output_size])
      for e in enc_out
      ]
    attention_states = tf.concat(top_states, 1)

    self.logger.debug('Initialize decoder')
    # ############################################################################
    # # Code from renzhe0009 @ StackOverflow
    # # http://stackoverflow.com/q/42703140/7390416
    # # License: MIT
    # # Because published after March 2016 - meta.stackexchange.com/q/272956
    #
    # # enc_state_c = tf.concat(values=(enc_state_fw.c, enc_state_bw.c), axis=1)
    # # enc_state_h = tf.concat(values=(enc_state_fw.h, enc_state_bw.h), axis=1)
    # enc_state_c = enc_state_fw.c + enc_state_bw.c
    # enc_state_h = enc_state_fw.h + enc_state_bw.h
    #
    # enc_state = LSTMStateTuple(c=enc_state_c, h=enc_state_h)
    # ############################################################################

    dec_out, dec_state = attention_decoder(
        self.decoder_inputs, enc_state_fw, cell=dec_cell,
        attention_states=attention_states, loop_function=loop_function
        )

    # Apply sigmoid activation to decoder outputs
    dec_out = tf.sigmoid(dec_out)

    # print('dec_state shape: ', dec_state[0].get_shape())
    # merge outputs into a tensor and transpose to be [B, seq_length, out_dim]
    dec_outputs = tf.transpose(tf.stack(dec_out), [1, 0, 2])
    # print('dec outputs shape: ', dec_outputs.get_shape())
    self.logger.info('dec outputs shape: {}'.format(dec_outputs.get_shape()))

    # Decoder outputs summaries
    split_dec_out = tf.split(dec_outputs, self.parameters_length, axis=2,
                             name='decoder_parameter')
    [self.decoder_outputs_summaries.append(
        histogram_summary(split_tensor.name, split_tensor)) for split_tensor in
      split_dec_out]

    # Separate decoder output into parameters and flags
    # params_in, flags_in = tf.split(dec_outputs, [42, 2], axis=2)

    self.encoder_vars = {}
    for tvar in tf.trainable_variables():
      if 'char_embedding' in tvar.name or 'encoder' in tvar.name:
        self.encoder_vars[tvar.name] = tvar
      print('tvar: ', tvar.name)

    return dec_outputs

  def save(self, sess, save_filename, global_step=None):
    if not hasattr(self, 'saver'):
      self.saver = tf.train.Saver()
    if not hasattr(self, 'encoder_saver'):
      self.encoder_saver = tf.train.Saver(var_list=self.encoder_vars)
    print('Saving checkpoint...')
    if global_step is not None:
      self.encoder_saver.save(sess, save_filename + '.encoder',
                              global_step)
      self.saver.save(sess, save_filename, global_step)
    else:
      self.encoder_saver.save(sess, save_filename + '.encoder')
      self.saver.save(sess, save_filename)

  def load(self, sess, save_path):
    if not hasattr(self, 'saver'):
      self.saver = tf.train.Saver()
    if os.path.exists(os.path.join(save_path, 'best_model.ckpt')):
      ckpt_name = os.path.join(save_path, 'best_model.ckpt')
      print('Loading checkpoint {}...'.format(ckpt_name))
      self.saver.restore(sess, os.path.join(ckpt_name))
    else:
      ckpt = tf.train.get_checkpoint_state(save_path)
      if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('Loading checkpoint {}...'.format(ckpt_name))
        self.saver.restore(sess, os.path.join(save_path, ckpt_name))
        return True
      else:
        return False



