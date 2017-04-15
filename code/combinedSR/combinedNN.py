from __future__ import print_function

import warnings
from time import gmtime, strftime

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting

logger_combined = logging.getLogger('combined')
logger_combined.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
formatter2 = logging.Formatter('%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_combined.addHandler(ch)

# File logger: see below META VARIABLES

import time
program_start_time = time.time()

print("\n * Importing libraries...")
from combinedNN_tools import *
from general_tools import *


##### SCRIPT META VARIABLES #####
VERBOSE = True
compute_confusion = False  # TODO: ATM this is not implemented

batch_size_audio = 1
batch_size_lip = 4
num_epochs = 50

nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons
N_HIDDEN_LIST = [64,64]
MAX_SEQ_LENGTH = 1000

BIDIRECTIONAL = True

# Decaying LR
LR_start = 0.01
logger_combined.info("LR_start = %s", str(LR_start))
LR_fin = 0.0000001
logger_combined.info("LR_fin = %s", str(LR_fin))
LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
#LR_decay= 0.5
logger_combined.info("LR_decay = %s", str(LR_decay))

#############################################################
# Set locations for DATA, LOG, PARAMETERS, TRAIN info
dataset = "combined"
root = os.path.expanduser("~/TCDTIMIT/combinedSR/")
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
logger_combined.addHandler(fh)
#############################################################


logger_combinedtools.info("\n\n\n\n STARTING NEW TRAINING SESSION AT " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

##### IMPORTING DATA #####

logger_combined.info('  data source: ' + data_path)
logger_combined.info('  model target: ' + model_save + '.npz')

dataset = unpickle(data_path)
X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset
# these are lists of np arrays, because the time sequences are different for each example
# X shape: (example, time_sequence, mfcc_feature,)
# y shape: (example, time_sequence,)


# Print some information
logger_combined.info("\n* Data information")
logger_combined.info('X train')
logger_combined.info('  %s %s', type(X_train), len(X_train))
logger_combined.info('  %s %s', type(X_train[0]), X_train[0].shape)
logger_combined.info('  %s %s', type(X_train[0][0]), X_train[0][0].shape)
logger_combined.info('  %s', type(X_train[0][0][0]))

logger_combined.info('y train')
logger_combined.info('  %s %s', type(y_train), len(y_train))
logger_combined.info('  %s %s', type(y_train[0]), y_train[0].shape)
logger_combined.info('  %s %s', type(y_train[0][0]), y_train[0][0].shape)

logger_combined.info('valid_frames train')
logger_combined.info('  %s %s', type(valid_frames_train), len(valid_frames_train))
logger_combined.info('  %s %s', type(valid_frames_train[0]), valid_frames_train[0].shape)
logger_combined.info('  %s %s', type(valid_frames_train[0][0]), valid_frames_train[0][0].shape)

##### BUIDING MODEL #####
logger_combined.info('\n* Building network ...')
RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size_audio, num_features=nbMFCCs, n_hidden_list=N_HIDDEN_LIST,
                            num_output_units=nbPhonemes, bidirectional=BIDIRECTIONAL, addDenseLayers=ADD_DENSE_LAYERS, seed=0, debug=False)
# print number of parameters
nb_params = lasagne.layers.count_params(RNN_network.network_output_layer)
logger_combined.info(" Number of parameters of this network: %s", nb_params)

# Try to load stored model
logger_combined.info(' Network built. Trying to load stored model: %s', model_load)
RNN_network.load_model(model_load)

##### COMPILING FUNCTIONS #####
logger_combined.info("\n* Compiling functions ...")
RNN_network.build_functions(train=True, debug=False)

##### TRAINING #####
logger_combined.info("\n* Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                  batch_size_audio=batch_size_audio, batch_size_lip=batch_size_lip, LR_start=LR_start, LR_decay=LR_decay,
                  compute_confusion=False, debug=False)

logger_combined.info("\n* Done")
logger_combined.info('Total time: {:.3f}'.format(time.time() - program_start_time))
