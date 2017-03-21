# import numpy as np
# import tensorflow.contrib.legacy_seq2seq as tf_seq2seq
# from tensorflow.contrib.rnn.python.ops import core_rnn, core_rnn_cell
#
#
# def decoder_loop_function(prev, i):
#   # TODO Change Placeholder call for actual function
#   return prev
#
#
# cell = core_rnn_cell.LSTMCell(256)
#
#
# # Try encapsulating encoder & decoder in functions, to be run in a Lambda layer
# def tf_seq2seq_encoder(encoder_inputs):
#   _, enc_state = core_rnn.static_rnn(
#     cell,
#     encoder_inputs,
#     dtype=float
#   )
#
#   return enc_state
#
#
# # Initialize Keras input
# # Initialize decoder input (loop_function?)
# enc_state_f = tf_seq2seq_encoder(np.zeros((1, 5, 5)))
#
#
# def tf_seq2seq_attention_decoder(decoder_inputs):
#   (outputs, state) = tf_seq2seq.attention_decoder(
#     decoder_inputs,
#     enc_state_f,
#     cell,
#     loop_function=decoder_loop_function
#   )
#
#   return outputs
#
#
# (dec_output, dec_state) = tf_seq2seq_attention_decoder(np.zeros((1, 5)))
#
################################################################################

# import tensorflow as tf
#
# # Model parameters
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1, 2, 3, 4]
# y_train = [0, -1, -2, -3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)  # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x: x_train, y: y_train})
#
# # evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
#
# # Build a dataflow graph.
# c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
# e = tf.matmul(c, d)
#
# # Construct a `Session` to execute the graph.
# sess = tf.Session()
#
# # Execute the graph and store the value that `e` represents in `result`.
# result = sess.run(e)
#
# ################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as tf_s2s
from tensorflow.contrib.rnn.python import ops as tf_rnn
from six.moves import xrange  # pylint: disable=redefined-builtin

# TODO Input these variables as TF Tensors
batch_size = 2
timesteps = 5

# tensor_enc = tf.placeholder(
#   dtype=tf.float32,
#   shape=(batch_size, timesteps, 4),
#   name='encoder_in')
# tensor_dec = tf.placeholder(
#   dtype=tf.float32,
#   shape=(batch_size, timesteps, 3),
#   name='decoder_in')
tensor_enc = [tf.placeholder(
  dtype=tf.float32,shape=(timesteps, 4)) for _ in xrange(batch_size)]
tensor_dec = [tf.placeholder(
  dtype=tf.float32,shape=(timesteps, 3)) for _ in xrange(batch_size)]

# params_enc = np.zeros((4, 1))  # [0] * 4
# params_dec = np.zeros((3, 1))  # [0] * 3
#
# frames_enc = [params_enc] * timesteps
# frames_dec = [params_dec] * timesteps
#
# seqs_enc = [frames_enc] * batch_size
# seqs_dec = [frames_dec] * batch_size
seqs_enc = [np.random.rand(timesteps, 4) for _ in xrange(batch_size)]
seqs_dec = [np.random.rand(timesteps, 3) for _ in xrange(batch_size)]

s2s = tf_s2s.basic_rnn_seq2seq(
  tensor_enc,
  tensor_dec,
  tf_rnn.core_rnn_cell.LSTMCell(256)
)
with tf.Session() as sess:
  # feed_dict = {tensor_enc:seqs_enc, tensor_dec: seqs_dec}
  feed_dict = {}
  for i, d in zip(tensor_enc, seqs_enc):
    feed_dict[i] = d
  for i, d in zip(tensor_dec, seqs_dec):
    feed_dict[i] = d

  init = tf.global_variables_initializer()
  sess.run(init)
  basic_seq2seq = sess.run(s2s, feed_dict=feed_dict) # tensor_enc: seqs_enc, tensor_dec: seqs_dec})

  print(basic_seq2seq)