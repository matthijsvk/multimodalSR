from __future__ import print_function

import warnings
from time import gmtime, strftime

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting

logger_RNN = logging.getLogger('audioSR')
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

# batch sizes: see just above training loop
num_epochs = 20

nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons
MANY_N_HIDDEN_LISTS = [[8], [8, 8], [8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8],
                      [32],[32,32],[32,32,32,32],
                       [64], [64,64],[64,64,64,64],
                       [256],[256,256],[256,256,256],
                       [512],[512,512],[512,512,512]]
MANY_N_HIDDEN_LISTS = [[1024], [1024, 1024], [1024, 1024, 1024, 1024]]

# Selected:
# MANY_N_HIDDEN_LISTS = [32,32],[64,64],[256,256],[512,512]
## for nbMFCC, uni vs bidirectional etc comparison:
#MANY_N_HIDDEN_LISTS = [[64,64]]


BIDIRECTIONAL = True
ADD_DENSE_LAYERS = False

justTest = False


for N_HIDDEN_LIST in MANY_N_HIDDEN_LISTS:
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
    dataset = "TIMIT" #""combined"
    test_dataset = "TIMIT";  #if justTestTrue, don't save the model, just run once over the TEST dataset
    root = os.path.expanduser("~/TCDTIMIT/audioSR/")
    store_dir = root + dataset + "/results"
    if not os.path.exists(store_dir): os.makedirs(store_dir)


    dataDir =  root + dataset + "/binary" + str(nbPhonemes) + os.sep + dataset  # output dir from datasetToPkl.py
    data_path = os.path.join(dataDir, dataset + '_' + str(nbMFCCs) + '_ch.pkl');
    if justTest:
        test_dataDir = root + test_dataset + "/binary" + str(nbPhonemes) + os.sep + test_dataset
        test_data_path = os.path.join(test_dataDir, test_dataset + '_' + str(nbMFCCs) + '_ch.pkl');
        logger_RNN.info("EVALUATING ON: %s", test_data_path)


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

    logger_RNN.info('  data source: %s', data_path)
    logger_RNN.info('  model target: %s', model_save + '.npz')

    dataset = unpickle(data_path)
    X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset
    # these are lists of np arrays, because the time sequences are different for each example
    # X shape: (example, time_sequence, mfcc_feature,)
    # y shape: (example, time_sequence,)
    if justTest: _,_,_,_,_,_,X_test,y_test, valid_frames_test = unpickle(test_data_path)
    dataset = [X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test]

    # Print some information
    debug=False
    if debug:
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
    if N_HIDDEN_LIST[0] > 128: batch_sizes = [32,16,8,4]
    else: batch_sizes = [128,64,32,16,8,4]
    for batch_size in batch_sizes:
        try:
            logger_RNN.info('\n* Building network using batch size: %s...', batch_size)
            RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size,
                                        num_features=nbMFCCs, n_hidden_list=N_HIDDEN_LIST,
                                        num_output_units=nbPhonemes,
                                        bidirectional=BIDIRECTIONAL, addDenseLayers=ADD_DENSE_LAYERS,
                                        debug=False)

            # print number of parameters
            nb_params = lasagne.layers.count_params(RNN_network.network_lout_batch)
            logger_RNN.info(" Number of parameters of this network: %s", nb_params)

            # Try to load stored model
            logger_RNN.info(' Network built. Trying to load stored model: %s', model_load)
            success = RNN_network.load_model(model_load)
            if success == 0: LR_start = LR_start / 10.0

            ##### COMPILING FUNCTIONS #####
            logger_RNN.info("\n* Compiling functions ...")
            RNN_network.build_functions(train=True, debug=False)

            ##### TRAINING #####
            logger_RNN.info("\n* Training ...")
            RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                              batch_size=batch_size, LR_start=LR_start, LR_decay=LR_decay,
                              compute_confusion=False, justTest=justTest, test_dataset=test_dataset,debug=False)
            break;
        except:
            print('caught this error: ' + traceback.format_exc());
            ##import pdb;pdb.set_trace()
            logger_RNN.info("batch size too large; trying again with lower batch size")
            pass  #just try again with the next batch_size
        fh.close()
        logger_RNN.removeHandler(fh)

logger_RNN.info("\n* Done")
logger_RNN.info('Total time: %s',time.time() - program_start_time)
