import numpy as np
# import coloredlogs, logging
# logger = logging.getLogger('your-module')
# coloredlogs.install(level='DEBUG')
# coloredlogs.install(level='DEBUG', logger=logger)
# #
# # a = np.array([[1, 2, 3],
# #               [3, 4, 5]])
# # b = np.array([[4, 5, 6],
# #               [1, 6, 7],
# #               [2, 7, 1]])
# # c = [a,b]
# # d = np.array([a,b])
# #
# #
# # #
# # # e =np.dstack([np.reshape(a,(1,a.shape[0],-1)), np.reshape(b, (1,b.shape[0], -1))])
# # #
# # print(a, a.shape)
# # print(b,b.shape)
# # # print(c,len(c))
# # #
# # # print(d, d.shape)
# # # print(D, D.shape)
# # #
# # # print(e,e.shape)
# #
# #
# # A = np.reshape(a, (1, a.shape[0], -1))
# # B = np.reshape(b, (1, b.shape[0], -1))
# # print(A, A.shape)
# # print(B, B.shape)
# # C = np.vstack([A, B])
# # print(C, C.shape)
#
# x = [ [[1, 2], [1, 2, 3], [1]], [[1, 2], [1, 2, 3], [2],[1]], [[1, 2], [1, 2, 3], [1]], [[1, 2], [1, 2, 3], [1]] ]
# y = np.array([np.array([bla for bla in xi]) for xi in x])
#
# Y = np.array(x)
# print(x,len(x), len(x[0]), len(x[0][0]))
# print(y,y.shape)
# print(Y, Y.shape)

# import numpy as np
# import random
# from six.moves import range
#
# def pad_sequences(sequences, maxlen=None, dtype='int32',
#                   padding='post', truncating='pre', value=0.):
#     """
#         Pad each sequence to the same length:
#         the length of the longuest sequence.
#         If maxlen is provided, any sequence longer
#         than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
#         the end of the sequence.
#         Supports post-padding and pre-padding (default).
#     """
#     lengths = [len(s) for s in sequences]
#
#     nb_samples = len(sequences)
#     if maxlen is None:
#         maxlen = np.max(lengths)
#     x = (np.ones((nb_samples, maxlen, sequences[0].shape[-1])) * value).astype(dtype)
#     for idx, s in enumerate(sequences):
#         if truncating == 'pre':
#             trunc = s[-maxlen:]
#         elif truncating == 'post':
#             trunc = s[:maxlen]
#         else:
#             raise ValueError("Truncating type '%s' not understood" % padding)
#
#         if padding == 'post':
#             x[idx, :len(trunc), :] = trunc
#         elif padding == 'pre':
#             x[idx, -len(trunc):, :] = np.array(trunc, dtype='float32')
#         else:
#             raise ValueError("Padding type '%s' not understood" % padding)
#     return x
# def generate_masks(inputs, batch_size):  # inputs = X
#     ## all recurrent layers in lasagne accept a separate mask input which has shape
#     # (batch_size, n_time_steps), which is populated such that mask[i, j] = 1 when j <= (length of sequence i) and mask[i, j] = 0 when j > (length
#     # of sequence i). When no mask is provided, it is assumed that all sequences in the minibatch are of length n_time_steps.
#     max_input_length = max([len(inputs[i]) for i in range(len(inputs))])
#     input_dim = len(inputs[0][0])
#     X = np.zeros((batch_size, max_input_length, input_dim))
#     input_mask = np.zeros((batch_size, max_input_length))
#
#     for example_id in range(len(inputs)):
#         curr_seq_len = len(inputs[example_id])
#         X[example_id, :curr_seq_len] = inputs[example_id]
#         input_mask[example_id, :curr_seq_len] = 1
#
#     return input_mask
# def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
#     """
#     Helper function that returns an iterator over the training data of a particular
#     size, optionally in a random order.
#
#     For big data sets you can load numpy arrays as memory-mapped files
#         (numpy.load(..., mmap_mode='r'))
#
#     This function a slight modification of:
#         http://lasagne.readthedocs.org/en/latest/user/tutorial.html
#     """
#     assert len(inputs) == len(targets)
#     if len(inputs) < batch_size:
#         batch_size = len(inputs)
#
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#
#     for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
#         print(start_idx)
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batch_size]
#         else:
#             excerpt = range(start_idx, start_idx + batch_size, 1)
#
#         input_iter = [inputs[i] for i in excerpt]
#         target_iter = [targets[i] for i in excerpt]
#         mask_iter = generate_masks(input_iter, batch_size)
#         seq_lengths = np.sum(mask_iter, axis=1)
#         yield input_iter, target_iter, mask_iter, seq_lengths
#
# batch_size = 16
#
# X = np.array([np.array([[1, 2, 4], [1, 2, 3], [1, 8, 2], [8, 5, 2], [1, 9, 4]]),
#               np.array([[1, 7, 2], [9, 4, 1]]),
#               np.array([[1, 5, 2], [1, 7, 3], [1, 4, 5], [3, 4, 5], [8, 3, 4]]),
#               np.array([[1, 5, 2], [1, 7, 3], [1, 4, 5], [3, 4, 5]])])
# y = np.array([np.array([bla for bla in xi]) for xi in X])
#
# print(X.shape, X[0].shape)
# print(X)
#
# m = generate_masks(X, batch_size)
# print("MINIBATCH...\n")
#
# for inputs, targets, masks, seq_lengths in iterate_minibatches(X, y, batch_size, shuffle=False):
#     print("########################################################")
#     inputs = np.array(inputs); targets = np.array(targets); masks = np.array(masks)
#     print("  INPUTS: ",inputs, inputs.shape)
#     print("  TARGETS: ",targets, targets.shape)
#     print("  MASKS: ",masks, masks.shape)
#     print("  SEQ_LEN: ", seq_lengths, seq_lengths.shape)
#
#     print("######################################################")
#     print("After padding: ")
#     X = pad_sequences(inputs)
#     y = pad_sequences(targets)
#     print("  INPUTS: ", X, X.shape)
#     print("  TARGETS: ", y, y.shape)


# LOGGING stuff
import logging

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')

# from https://docs.python.org/3/howto/logging-cookbook.html
import logging, colorFormatting  # debug < info < warn < error < critical

# You need to change the default format in colorFormatting.py (at the bottom) to CRITICAL
# otherwise it will print all Python/ Theano debug messages as well
logging.setLoggerClass(colorFormatting.ColoredLogger)
logger_RNN = logging.getLogger('RNN')
logger_RNN.setLevel(logging.DEBUG)






