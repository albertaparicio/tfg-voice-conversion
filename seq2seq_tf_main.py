# Created by albert aparicio on 02/04/17
# coding: utf-8

# This script defines an Sequence-to-Sequence model, using the implemented
# encoders and decoders from TensorFlow

# TODO Document and explain steps

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import argparse
import gzip
import logging
import os
import timeit
from sys import version_info
import numpy as np
import tensorflow as tf

from seq2seq_tf_model import Seq2Seq, DataLoader

# Conditional imports
if version_info.major > 2:
  import pickle
else:
  import cPickle as pickle

# Initialize logger at INFO level
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def main(args):
  # If-else for training and testing
  if args.do_train:
    # Load training dataset
    # (src_datatable, trg_datatable, trg_masks, train_spk, train_spk_max,
    # train_spk_min, max_seq_length) = load_dataset(args.train_data_path,
    #                                                    args.train_out_file,
    #                                                    args.save_h5,
    #                                                    args.val_fraction)

    dl = DataLoader(args)

    # TODO Remove '10' and put max_seq_length in its place
    seq2seq_model = Seq2Seq(args.enc_rnn_layers, args.dec_rnn_layers,
                            # args.rnn_size, 10, args.params_len)
                            args.rnn_size, dl.max_seq_length, args.params_len)

    train(args, seq2seq_model, dl)

  if args.do_test:
    tf.reset_default_graph()

    # Load test dataset
    (src_test_datatable, src_test_masks, trg_test_datatable, trg_test_masks,
     test_speakers, test_speakers_max, test_speakers_min,
     max_test_length) = load_dataset(args.test_data_path, args.test_out_file,
                                     args.save_h5, )

    seq2seq_model = Seq2Seq(args.enc_rnn_layers, args.dec_rnn_layers,
                            args.rnn_size, max_test_length, args.params_len)

    test(args, seq2seq_model)


def train(args, model, dl):
  with tf.Session() as sess:
    batch_idx = 0
    curr_epoch = 0
    count = 0
    batch_timings = []
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

    for src_batch, trg_batch, trg_mask in dl.next_batch():
      if curr_epoch == 0 and batch_idx == 0:
        batches_per_epoch = dl.batches_per_epoch
        logging.info('Batches per epoch: {}'.format(batches_per_epoch))
      beg_t = timeit.default_timer()

      # Model's placeholders
      feed_dict = {
        model.gtruth: trg_batch[:, 0:int(model.gtruth.get_shape()[1]), :]}

      for i, enc_in in enumerate(model.encoder_inputs):
        # TODO Fix encoder_inputs so that it takes 64 parameters as input
        # feed_dict[enc_in] = src_batch[:,i,:]
        feed_dict[enc_in] = src_batch[:, i, 0:44]

      # Decoder inputs are trg_batch with the first timestep set to 0 (GO)
      trg_batch = np.roll(trg_batch, 1, axis=1)
      trg_batch[:, 0, :] = np.zeros((model.batch_size, model.parameters_length))

      for i, dec_in in enumerate(model.decoder_inputs):
        feed_dict[dec_in] = trg_batch[:, i, :]

      # tr_loss, _, enc_state, summary = sess.run(
      #   [model.loss, model.train_op, model.enc_state, merged],
      #   feed_dict=feed_dict)
      tr_loss, _ = sess.run(
        [model.loss, model.train_op],
        feed_dict=feed_dict)

      batch_timings.append(timeit.default_timer() - beg_t)

      # TODO Use the right variables in this print
      logging.info('{}/{} (epoch {}) loss {}, time/batch {} s'.format(
        count,
        opts.epoch * batches_per_epoch,
        curr_epoch,
        tr_loss,
        np.mean(batch_timings)))
      if batch_idx % opts.save_every == 0:
        # train_writer.add_summary(summary, count)
        checkpoint_file = os.path.join(opts.save_path, 'model.ckpt')
        model.save(sess, checkpoint_file, count)

      batch_idx += 1
      count += 1
      if curr_epoch >= opts.epoch:
        break

  return


def evaluate(sess, args, model):
  """ Evaluate an epoch over the given data_loader batches """

  # TODO Cridar run sols amb model.loss, sense train_op
  return


def test(args, model):
  """ Evaluate an epoch over the given data_loader batches """
  return


if __name__ == '__main__':
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
  parser.add_argument('--no-train', dest='do_train',
                      action='store_false', help='Flag to train or not.')
  parser.add_argument('--no-test', dest='do_test',
                      action='store_false', help='Flag to test or not.')
  parser.add_argument('--save_path', type=str, default="training_results")
  parser.set_defaults(do_train=True, do_test=True, save_h5=False)
  opts = parser.parse_args()

  if not os.path.exists(opts.save_path):
    os.makedirs(opts.save_path)
  # save config
  with gzip.open(os.path.join(opts.save_path, 'config.pkl.gz'), 'wb') as cf:
    pickle.dump(opts, cf)

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
