# Created by albert aparicio on 31/03/17
# coding: utf-8

# This script defines a data loader for the Seq2Seq model

# TODO Document and explain steps
# TODO Move this model to tfglib

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np
import tfglib.seq2seq_datatable as s2s
from tfglib.seq2seq_normalize import maxmin_scaling
from tfglib.utils import init_logger


class DataLoader(object):
  # TODO Finish this class and move it to a new file
  def __init__(self, args, test=False, max_seq_length=None, shortseq=True,
               logger_level='INFO'):
    self.logger = init_logger(name=__name__, level=logger_level)

    self.logger.debug('DataLoader init')
    self.batch_size = args.batch_size

    if test:
      self.s2s_datatable = s2s.Seq2SeqDatatable(
          args.test_data_path, args.test_out_file,
          basenames_file='tcstar_basenames.list', shortseq=shortseq,
          max_seq_length=int(max_seq_length), vocoded_dir='tcstar_vocoded')

      (self.src_test_data, self.src_seq_len, self.trg_test_data,
       self.trg_test_masks_f, self.trg_seq_len, self.train_src_speakers,
       self.train_src_speakers_max, self.train_src_speakers_min,
       self.train_trg_speakers, self.train_trg_speakers_max,
       self.train_trg_speakers_min, dataset_max_seq_length) = self.load_dataset(
          args.train_out_file,
          args.save_h5,
          test=test
          )

      self.test_batches_per_epoch = int(
          np.floor(self.src_test_data.shape[0] / self.batch_size)
          )

    else:
      self.s2s_datatable = s2s.Seq2SeqDatatable(
          args.train_data_path, args.train_out_file,
          basenames_file='tcstar_basenames.list', shortseq=shortseq,
          max_seq_length=int(max_seq_length), vocoded_dir='tcstar_vocoded')

      (src_datatable, self.src_seq_len, trg_datatable, trg_masks,
       self.trg_seq_len, self.train_src_speakers, self.train_src_speakers_max,
       self.train_src_speakers_min, self.train_trg_speakers,
       self.train_trg_speakers_max, self.train_trg_speakers_min,
       dataset_max_seq_length) = self.load_dataset(
          args.train_out_file,
          args.save_h5
          )

      self.logger.debug('Split into training and validation')
      ################################################
      # Split data into training and validation sets #
      ################################################
      # ############################
      # # COMMENT AFTER DEVELOPING #
      # ############################
      # batch_size = 2
      # nb_epochs = 2
      #
      # num = 10
      # src_datatable = src_datatable[0:num]
      # src_masks = src_masks[0:num]
      # trg_datatable = trg_datatable[0:num]
      # trg_masks = trg_masks[0:num]
      #
      # model_description = 'DEV_' + model_description
      # #################################################

      self.src_train_data = src_datatable[0:int(np.floor(
          src_datatable.shape[0] * (1 - args.val_fraction)))]
      self.src_valid_data = src_datatable[int(np.floor(
          src_datatable.shape[0] * (1 - args.val_fraction))):]

      self.trg_train_data = trg_datatable[0:int(np.floor(
          trg_datatable.shape[0] * (1 - args.val_fraction)))]
      self.trg_valid_data = trg_datatable[int(np.floor(
          trg_datatable.shape[0] * (1 - args.val_fraction))):]

      self.trg_train_masks_f = trg_masks[0:int(np.floor(
          trg_masks.shape[0] * (1 - args.val_fraction)))]
      self.trg_valid_masks_f = trg_masks[int(np.floor(
          trg_masks.shape[0] * (1 - args.val_fraction))):]

      self.train_batches_per_epoch = int(
          np.floor(self.src_train_data.shape[0] / self.batch_size)
          )
      self.valid_batches_per_epoch = int(
          np.floor(self.src_valid_data.shape[0] / self.batch_size)
          )

    if shortseq:
      self.max_seq_length = max_seq_length
    else:
      self.max_seq_length = dataset_max_seq_length

  def load_dataset(self, train_out_file, save_h5, test=False):
    import h5py

    self.logger.debug('Load test dataset')
    if save_h5:
      self.logger.info('Saving datatable')

      (src_datatable,
       src_masks,
       src_seq_len,
       trg_datatable,
       trg_masks,
       trg_seq_len,
       train_src_speakers_max,
       train_src_speakers_min,
       train_trg_speakers_max,
       train_trg_speakers_min
       ) = self.s2s_datatable.seq2seq_save_datatable()

      self.logger.info('DONE - Saving datatable')

    else:
      self.logger.info('Load parameters')
      (src_datatable,
       src_masks,
       src_seq_len,
       trg_datatable,
       trg_masks,
       trg_seq_len,
       train_src_speakers_max,
       train_src_speakers_min,
       train_trg_speakers_max,
       train_trg_speakers_min
       ) = self.s2s_datatable.seq2seq_load_datatable()
      self.logger.info('DONE - Loaded parameters')

    if test:
      # Load training speakers data
      with h5py.File(train_out_file + '.h5', 'r') as file:
        # Load datasets
        train_src_speakers_max = file.attrs.get('src_speakers_max')
        train_src_speakers_min = file.attrs.get('src_speakers_min')
        train_trg_speakers_max = file.attrs.get('trg_speakers_max')
        train_trg_speakers_min = file.attrs.get('trg_speakers_min')

        file.close()

    train_src_speakers = train_src_speakers_max.shape[0]
    train_trg_speakers = train_trg_speakers_max.shape[0]

    # Normalize data
    self.logger.debug('Normalize data')

    # Iterate over sequence 'slices'
    assert src_datatable.shape[0] == trg_datatable.shape[0]

    for i in range(src_datatable.shape[0]):
      (
        src_datatable[i, :, 0:42],
        trg_datatable[i, :, 0:42]
        ) = maxmin_scaling(
          src_datatable[i, :, :],
          src_masks[i, :],
          trg_datatable[i, :, :],
          trg_masks[i, :],
          train_src_speakers_max,
          train_src_speakers_min,
          train_trg_speakers_max,
          train_trg_speakers_min
          )

    return (src_datatable, src_seq_len, trg_datatable, trg_masks, trg_seq_len,
            train_src_speakers, train_src_speakers_max, train_src_speakers_min,
            train_trg_speakers, train_trg_speakers_max, train_trg_speakers_min,
            self.s2s_datatable.max_seq_length)

  def next_batch(self, test=False, validation=False):
    self.logger.debug('Choice between training data or validation data')
    if test:
      data_dict = {'src_data'         : self.src_test_data,
                   'trg_data'         : self.trg_test_data,
                   'trg_mask'         : self.trg_test_masks_f,
                   'batches_per_epoch': self.test_batches_per_epoch}
    else:
      if validation:
        data_dict = {'src_data'         : self.src_valid_data,
                     'trg_data'         : self.trg_valid_data,
                     'trg_mask'         : self.trg_valid_masks_f,
                     'batches_per_epoch': self.valid_batches_per_epoch}
      else:
        # Training
        data_dict = {'src_data'         : self.src_train_data,
                     'trg_data'         : self.trg_train_data,
                     'trg_mask'         : self.trg_train_masks_f,
                     'batches_per_epoch': self.train_batches_per_epoch}

    self.logger.debug('Initialize next batch generator')
    batch_id = 0

    while True:
      self.logger.debug('--> Next batch - Prepare <--')
      src_batch = data_dict['src_data'][
                  batch_id * self.batch_size:(batch_id + 1) * self.batch_size,
                  :, :]
      src_batch_seq_len = self.src_seq_len[
                          batch_id * self.batch_size:
                          (batch_id + 1) * self.batch_size]
      trg_batch = data_dict['trg_data'][
                  batch_id * self.batch_size:(batch_id + 1) * self.batch_size,
                  :, :]
      trg_mask = data_dict['trg_mask'][
                 batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :]

      batch_id = (batch_id + 1) % data_dict['batches_per_epoch']

      self.logger.debug('--> Next batch - Yield <--')
      yield (src_batch, src_batch_seq_len, trg_batch, trg_mask)
