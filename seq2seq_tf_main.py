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
from sys import version_info

import tensorflow as tf
import tfglib.seq2seq_datatable as s2s

from seq2seq_tf_model import Seq2Seq

# Conditional imports
if version_info.major > 2:
  import pickle
else:
  import cPickle as pickle

# Initialize logger at INFO level
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_dataset(data_path, out_file, save_h5):
  if save_h5:
    logging.info('Saving training datatable')

    (src_train_datatable,
     src_train_masks,
     trg_train_datatable,
     trg_train_masks,
     max_train_length,
     train_speakers_max,
     train_speakers_min
     ) = s2s.seq2seq_save_datatable(data_path, out_file)

    logging.info('DONE Saving training datatable')

  else:
    print('Load pretraining parameters')
    (src_train_datatable,
     src_train_masks,
     trg_train_datatable,
     trg_train_masks,
     max_train_length,
     train_speakers_max,
     train_speakers_min
     ) = s2s.seq2seq2_load_datatable(opts.train_out_file + '.h5')

  train_speakers = train_speakers_max.shape[0]

  return (
    train_speakers, train_speakers_max, train_speakers_min, max_train_length
  )


def main(args):
  # If-else for training and testing
  if args.do_train:
    # Load training dataset
    (train_speakers, train_speakers_max, train_speakers_min, max_train_length
     ) = load_dataset(args.train_data_path, args.train_out_file, args.save_h5)

    seq2seq_model = Seq2Seq(args.enc_rnn_layers, args.dec_rnn_layers,
                            args.rnn_size, max_train_length, args.params_len)

    train(args, seq2seq_model)

  if args.do_test:
    tf.reset_default_graph()

    # Load test dataset
    (test_speakers, test_speakers_max, test_speakers_min, max_test_length
     ) = load_dataset(args.test_data_path, args.test_out_file, args.save_h5)

    seq2seq_model = Seq2Seq(args.enc_rnn_layers, args.dec_rnn_layers,
                            args.rnn_size, max_test_length, args.params_len)

    test(args, seq2seq_model)


def train(args, model):
  return


def evaluate(sess, args, model):
  """ Evaluate an epoch over the given data_loader batches """
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
  parser.add_argument('--save-h5', dest='save_h5', action='store_true',
                      help='Save dataset to .h5 file')
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
  parser.add_argument('--save_path', type=str, default="checkpoint")
  parser.set_defaults(do_train=True, do_test=True, save_h5=False)
  opts = parser.parse_args()

  if not os.path.exists(opts.save_path):
    os.makedirs(opts.save_path)
  # save config
  with gzip.open(os.path.join(opts.save_path, 'config.pkl.gz'), 'wg') as cf:
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
