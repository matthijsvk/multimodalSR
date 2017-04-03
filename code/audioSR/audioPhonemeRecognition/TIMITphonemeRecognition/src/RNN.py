from __future__ import print_function

import os, warnings
warnings.simplefilter("ignore", UserWarning)  #cuDNN warning

import logging, colorFormatting # debug < info < warn < error < critical  # from https://docs.python.org/3/howto/logging-cookbook.html
logging.setLoggerClass(colorFormatting.ColoredLogger)
logger_RNN = logging.getLogger('RNN')
logger_RNN.setLevel(logging.DEBUG)
# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
# ch.setFormatter(formatter)
# logger_RNN.addHandler(ch)


import time
program_start_time = time.time()

logger_RNN.info("\n * Importing libraries...")
from RNN_tools_lstm import *
from general_tools import *


logger_RNN.info('\n * Setting up ...')
##### SCRIPT META VARIABLES #####
VERBOSE = True
compute_confusion = False  # TODO: ATM this is not implemented

num_epochs = 20
batch_size = 8

INPUT_SIZE = 26  # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
NUM_OUTPUT_UNITS = 39
N_HIDDEN = 10

LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_INIT = 0.1

dataRootPath = "/home/matthijs/TCDTIMIT/TIMIT/binary_list39/speech2phonemes26Mels/"
data_path = dataRootPath + "std_preprocess_26_ch.pkl"
# train_data_path = os.path.join(dataRootPath, 'trainData.pkl')
# test_data_path = os.path.join(dataRootPath, 'testData.pkl')

output_path = "/home/matthijs/TCDTIMIT/TIMIT/binary/results"
model_name = "1HiddenLayer" + str(N_HIDDEN) + "_nbMFCC" + str(INPUT_SIZE)
model_load = os.path.join(output_path, model_name + 'oieno.npz')
model_save = os.path.join(output_path, model_name)


##### IMPORTING DATA #####
logger_RNN.info('  data source: ' + dataRootPath)
logger_RNN.info('  model target: ' + model_save + '.npz')

dataset = load_dataset(data_path)
X_train, y_train, X_val, y_val, X_test, y_test = dataset


# Print some information
if VERBOSE:
    logger_RNN.info("\n* Data information")
    logger_RNN.info('  X train')
    logger_RNN.info('%s %s', type(X_train), len(X_train))
    logger_RNN.info('%s %s',type(X_train[0]), X_train[0].shape)
    logger_RNN.info('%s %s',type(X_train[0][0]), X_train[0][0].shape)
    logger_RNN.info('%s',type(X_train[0][0][0]))

    logger_RNN.info('  y train')
    logger_RNN.info('%s %s',type(y_train), len(y_train))
    logger_RNN.info('%s %s',type(y_train[0]), y_train[0].shape)
    logger_RNN.info('%s %s',type(y_train[0][0]), y_train[0][0].shape)


##### BUIDING MODEL #####
logger_RNN.info('\n* Building network ...')
RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size, num_features=INPUT_SIZE, n_hidden=N_HIDDEN,
                            num_output_units=NUM_OUTPUT_UNITS, seed=0, debug=False)
logger_RNN.info(' Network built. Trying to load stored model...')
RNN_network.load_model(model_load)


##### BUIDING FUNCTION #####
logger_RNN.info("\n* Compiling functions ...")
RNN_network.build_functions(LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, debug=False)


##### TRAINING #####
logger_RNN.info("\n* Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                  batch_size=batch_size, compute_confusion=False, debug=True)


logger_RNN.info("\n* Done")
logger_RNN.info('Total time: {:.3f}'.format(time.time() - program_start_time))
