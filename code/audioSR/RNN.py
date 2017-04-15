from __future__ import print_function

import warnings
from time import gmtime, strftime

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting

logger_RNN = logging.getLogger('RNN')
logger_RNN.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
formatter2 = logging.Formatter('%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_RNN.addHandler(ch)

# File logger: see below META VARIABLES


import time
program_start_time = time.time()

print("\n * Importing libraries...")
from RNN_tools_lstm import *
from general_tools import *


##### SCRIPT META VARIABLES #####
VERBOSE = True
compute_confusion = False  # TODO: ATM this is not implemented

batch_size = 32
num_epochs = 50

nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons
N_HIDDEN_LIST = [64,64]
MAX_SEQ_LENGTH = 1000

BIDIRECTIONAL = True
ADD_DENSE_LAYERS = False

# Decaying LR
LR_start = 0.01
logger_RNN.info("LR_start = %s", str(LR_start))
LR_fin = 0.0000001
logger_RNN.info("LR_fin = %s", str(LR_fin))
# LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
LR_decay= 0.5
logger_RNN.info("LR_decay = %s", str(LR_decay))

#############################################################
# Set locations for DATA, LOG, PARAMETERS, TRAIN info
dataset = "combined"
root = os.path.expanduser("~/TCDTIMIT/audioSR/")
store_dir = root + dataset + "/results"
if not os.path.exists(store_dir): os.makedirs(store_dir)


dataDir = dataRootDir = root + dataset + "/binary" + str(nbPhonemes) + os.sep + dataset  # output dir from datasetToPkl.py
data_path = os.path.join(dataDir, dataset + '_' + str(nbMFCCs) + '_ch.pkl');

model_name = str(len(N_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join([str(layer) for layer in N_HIDDEN_LIST]) \
             + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if BIDIRECTIONAL else "_unidirectional") + \
("_withDenseLayers" if ADD_DENSE_LAYERS else "") + "_" + dataset


# model parameters and network_training_info
model_load = os.path.join(store_dir, model_name + ".npz")
model_save = os.path.join(store_dir, model_name)

# log file
logFile = store_dir + os.sep + model_name + '.log'
if os.path.exists(logFile):
    fh = logging.FileHandler(logFile)  # append to existing log
else:
    fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_RNN.addHandler(fh)
#############################################################


logger_RNNtools.info("\n\n\n\n STARTING NEW TRAINING SESSION AT " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

##### IMPORTING DATA #####

logger_RNN.info('  data source: ' + data_path)
logger_RNN.info('  model target: ' + model_save + '.npz')

dataset = load_dataset(data_path)
X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset
# these are lists of np arrays, because the time sequences are different for each example
# X shape: (example, time_sequence, mfcc_feature,)
# y shape: (example, time_sequence,)


# Print some information
logger_RNN.info("\n* Data information")
logger_RNN.info('X train')
logger_RNN.info('  %s %s', type(X_train), len(X_train))
logger_RNN.info('  %s %s', type(X_train[0]), X_train[0].shape)
logger_RNN.info('  %s %s', type(X_train[0][0]), X_train[0][0].shape)
logger_RNN.info('  %s', type(X_train[0][0][0]))

logger_RNN.info('y train')
logger_RNN.info('  %s %s', type(y_train), len(y_train))
logger_RNN.info('  %s %s', type(y_train[0]), y_train[0].shape)
logger_RNN.info('  %s %s', type(y_train[0][0]), y_train[0][0].shape)

logger_RNN.info('valid_frames train')
logger_RNN.info('  %s %s', type(valid_frames_train), len(valid_frames_train))
logger_RNN.info('  %s %s', type(valid_frames_train[0]), valid_frames_train[0].shape)
logger_RNN.info('  %s %s', type(valid_frames_train[0][0]), valid_frames_train[0][0].shape)

##### BUIDING MODEL #####
logger_RNN.info('\n* Building network ...')
RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size, num_features=nbMFCCs, n_hidden_list=N_HIDDEN_LIST,
                            num_output_units=nbPhonemes, bidirectional=BIDIRECTIONAL, addDenseLayers=ADD_DENSE_LAYERS, seed=0, debug=False)
# print number of parameters
nb_params = lasagne.layers.count_params(RNN_network.network_output_layer)
logger_RNN.info(" Number of parameters of this network: %s", nb_params)

# Try to load stored model
logger_RNN.info(' Network built. Trying to load stored model: %s', model_load)
RNN_network.load_model(model_load)

##### COMPILING FUNCTIONS #####
logger_RNN.info("\n* Compiling functions ...")
RNN_network.build_functions(train=True, debug=False)

##### TRAINING #####
logger_RNN.info("\n* Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                  batch_size=batch_size, LR_start=LR_start, LR_decay=LR_decay,
                  compute_confusion=False, debug=False)

logger_RNN.info("\n* Done")
logger_RNN.info('Total time: {:.3f}'.format(time.time() - program_start_time))
