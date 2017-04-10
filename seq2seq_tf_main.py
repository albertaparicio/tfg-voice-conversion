# Created by albert aparicio on 02/04/17
# coding: utf-8

# This script defines an Sequence-to-Sequence model, using the implemented
# encoders and decoders from TensorFlow

# TODO Document and explain steps

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import argparse
import gzip
import os
import timeit
from sys import version_info

import numpy as np
import tensorflow as tf
from tfglib.utils import init_logger

from seq2seq_tf_model import Seq2Seq, DataLoader

# Conditional imports
if version_info.major > 2:
  import pickle
else:
  import cPickle as pickle

# logger.debug('Before if __main__')
if __name__ == '__main__':
  # logger.debug('Before parsing args')
  parser = argparse.ArgumentParser(
    description="Convert voice signal with seq2seq model")
  parser.add_argument('--train_data_path', type=str, default="data/training/")
  parser.add_argument('--train_out_file', type=str,
                      default="data/seq2seq_train_datatable")
  parser.add_argument('--test_data_path', type=str, default="data/test/")
  parser.add_argument('--test_out_file', type=str,
                      default="data/seq2seq_test_datatable")
  parser.add_argument('--val_fraction', type=float, default=0.25)
  parser.add_argument('--save-h5', dest='save_h5', action='store_true',
                      help='Save dataset to .h5 file')
  parser.add_argument('--params_len', type=int, default=44)
  parser.add_argument('--patience', type=int, default=4,
                      help="Patience epochs to do validation, if validation "
                           "score is worse than train for patience epochs "
                           ", quit training. (Def: 4).")
  parser.add_argument('--enc_rnn_layers', type=int, default=1)
  parser.add_argument('--dec_rnn_layers', type=int, default=1)
  parser.add_argument('--rnn_size', type=int, default=256)
  parser.add_argument('--cell_type', type=str, default="lstm")
  parser.add_argument('--batch_size', type=int, default=20)
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument('--optimizer', type=str, default="adam")
  parser.add_argument('--clip_norm', type=float, default=5)
  parser.add_argument('--attn_length', type=int, default=500)
  parser.add_argument('--attn_size', type=int, default=100)
  parser.add_argument('--save_every', type=int, default=200)
  parser.add_argument('--no-train', dest='do_train',
                      action='store_false', help='Flag to train or not.')
  parser.add_argument('--no-test', dest='do_test',
                      action='store_false', help='Flag to test or not.')
  parser.add_argument('--save_path', type=str, default="training_results")
  parser.add_argument('--log', type=str, default="INFO")
  # TODO Add parameter to set logging level
  parser.set_defaults(do_train=True, do_test=True, save_h5=False)
  opts = parser.parse_args()

  # Initialize logger
  # logger_level = opts.log
  logger = init_logger(name=__name__, level=opts.log)

  logger.debug('Parsed arguments')
  if not os.path.exists(opts.save_path):
    os.makedirs(opts.save_path)
  # save config
  with gzip.open(os.path.join(opts.save_path, 'config.pkl.gz'), 'wb') as cf:
    pickle.dump(opts, cf)

logger.debug('Before defining main')


def main(args):
  logger.debug('Main')

  # If-else for training and testing
  if args.do_train:
    logger.info('Training')

    logger.debug('Initialize DataLoader')
    dl = DataLoader(args, max_seq_length=10, logger_level=opts.log)

    logger.debug('Initialize model')
    # TODO Remove '10' and put max_seq_length in its place
    seq2seq_model = Seq2Seq(opts.log, args.enc_rnn_layers, args.dec_rnn_layers,
                            # args.rnn_size, 10, args.params_len)
                            args.rnn_size, dl.max_seq_length, args.params_len)

    logger.info('Start training')
    train(args, seq2seq_model, dl)

  if args.do_test:
    tf.reset_default_graph()

    logger.debug('Initialize DataLoader')
    # Load test dataset
    dl = DataLoader(args, logger_level=opts.log)

    logger.debug('Initialize model')
    seq2seq_model = Seq2Seq(opts.log, args.enc_rnn_layers, args.dec_rnn_layers,
                            # args.rnn_size, 10, args.params_len)
                            args.rnn_size, dl.max_seq_length, args.params_len)

    test(args, seq2seq_model, dl)


logger.debug('Defined main')

logger.debug('Before define train')


def train(args, model, dl):
  logger.debug('Inside train')
  with tf.Session() as sess:
    logger.debug('Define constants')
    batch_idx = 0
    curr_epoch = 0
    count = 0
    batch_timings = []

    logger.debug('Initialize TF variables')
    try:
      tf.global_variables_initializer().run()
      merged = tf.summary.merge_all()
    except AttributeError:
      # Backward compatibility
      tf.initialize_all_variables().run()
      merged = tf.merge_all_summaries()
    LOG_DIR = os.path.join(opts.save_path, 'tf_train')
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    best_val_loss = 10e6
    curr_patience = opts.patience
    logger.debug('Defined constants')

    logger.debug('Start batch loop')
    for src_batch, trg_batch, trg_mask in dl.next_batch():
      if curr_epoch == 0 and batch_idx == 0:
        logger.info('Batches per epoch: {}'.format(dl.batches_per_epoch))
        logger.info(
          'Total batches: {}'.format(dl.batches_per_epoch * opts.epoch))
      beg_t = timeit.default_timer()

      # Model's placeholders
      logger.debug('Fill feed_dict with gtruth')
      feed_dict = {
        model.gtruth    : trg_batch[:, 0:int(model.gtruth.get_shape()[1]), :],
        model.seq_length: int(dl.max_seq_length) * np.ones((model.batch_size,),
                                                           dtype=np.int32)}

      logger.debug('feed_dict - encoder_inputs')
      for i, enc_in in enumerate(model.encoder_inputs):
        # TODO Fix encoder_inputs so that it takes 64 parameters as input
        # feed_dict[enc_in] = src_batch[:,i,:]
        feed_dict[enc_in] = src_batch[:, i, 0:44]

      # Decoder inputs are trg_batch with the first timestep set to 0 (GO)
      logger.debug('Roll decoder inputs and add GO symbol')
      trg_batch = np.roll(trg_batch, 1, axis=1)
      trg_batch[:, 0, :] = np.zeros((model.batch_size, model.parameters_length))

      logger.debug('feed_dict - decoder_inputs')
      for i, dec_in in enumerate(model.decoder_inputs):
        feed_dict[dec_in] = trg_batch[:, i, :]

      logger.debug('sess.run')
      # tr_loss, _, enc_state, summary = sess.run(
      #   [model.loss, model.train_op, model.enc_state, merged],
      #   feed_dict=feed_dict)
      tr_loss, _, enc_state = sess.run(
        [model.loss, model.train_op, model.enc_state],
        feed_dict=feed_dict)

      logger.debug('Append batch timings')
      batch_timings.append(timeit.default_timer() - beg_t)

      # Print batch info
      logger.info('Batch {}/{} (epoch {}) loss {}, time/batch (s) {}'.format(
        count,
        opts.epoch * dl.batches_per_epoch,  # Total num. of batches
        curr_epoch + 1,
        tr_loss,
        np.mean(batch_timings)))
      if batch_idx % opts.save_every == 0:
        # train_writer.add_summary(summary, count)
        logger.info('Save checkpoint')
        checkpoint_file = os.path.join(opts.save_path, 'model.ckpt')
        model.save(sess, checkpoint_file, count)
      if batch_idx >= dl.batches_per_epoch:
        curr_epoch += 1
        batch_idx = 0
        # eval

        # TODO Complete evaluate method
        # va_loss = np.mean(do_eval(sess, opts, model,
        #                           data_loader(), curr_epoch))
        # if va_loss < best_val_loss:
        #   print('Val loss improved {} --> {}'.format(best_val_loss,
        #                                              va_loss))
        #   # update the best score
        #   best_val_loss = va_loss
        #   curr_patience = opts.patience
        #   best_checkpoint_file = os.path.join(opts.save_path,
        #                                       'best_model.ckpt')
        #   model.save(sess, best_checkpoint_file)
        # else:
        #   print('Val loss did not improve, patience: ', curr_patience)
        #   curr_patience -= 1
        #   if model.optimizer == 'sgd':
        #     # if we have SGD optimizer, half the learning rate
        #     curr_lr = sess.run(model.curr_lr)
        #     print('Halving lr {} --> {} in SGD'.format(curr_lr,
        #                                                0.5 * curr_lr))
        #     sess.run(tf.assign(model.curr_lr, curr_lr * .5))
        #   if curr_patience == 0:
        #     print('Out of patience ({}) at epoch {} with '
        #           'tr_loss {} and '
        #           'best_val_loss {}'.format(opts.patience,
        #                                     curr_epoch,
        #                                     tr_loss,
        #                                     best_val_loss))
        #     break

      batch_idx += 1
      count += 1
      if curr_epoch >= opts.epoch:
        logger.info('Finished epochs -> BREAK')
        break


logger.debug('Defined train')

logger.debug('Before define evaluate')


def evaluate(sess, args, model):
  """ Evaluate an epoch over the given data_loader batches """

  # TODO Cridar run sols amb model.loss, sense train_op
  return


logger.debug('Defined evaluate')

logger.debug('Before define test')


def test(args, model, dl):
  """ Evaluate an epoch over the given data_loader batches """
  return


logger.debug('Defined test')

if __name__ == '__main__':
  logger.debug('Before calling main')
  main(opts)

  # Unused arguments
  #
  # parser.add_argument('--seq_len', type=int, default=500)
  # parser.add_argument('--params_len', type=int, default=500)
  # parser.add_argument('--test_words', type=str, default="data/embed_words")
  # parser.add_argument('--test_preds', type=str, default="test.preds",
  #                     help='File to store the predictions of test set.')
  # parser.add_argument('--test_words', type=str, default="lol")
  # parser.add_argument('--save_every', type=int, default=200)
  # # parser.add_argument('--max_word_len', type=int, default=26)
  # # parser.add_argument('--max_phoneme_len', type=int, default=57)
  # parser.add_argument('--highway_layers', type=int, default=2)
  # parser.add_argument('--chars_emb_dim', type=int, default=500)
  # parser.add_argument('--phones_emb_dim', type=int, default=500)
  # parser.add_argument('--vocab', type=str,
  #                     default='data/vocab_char2ph.pkl.gz',
  #                     help='Path to vocabs pickle (Default: '
  #                          'data/train_char2ph.pkl.gz).')
  # parser.add_argument('--out_prefix', type=str, default="word2ph")
  # parser.set_defaults(do_train=True, do_test=True, conv_frontend=False)
