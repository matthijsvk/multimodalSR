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
import preprocessingCombined

##### SCRIPT META VARIABLES #####
VERBOSE = True
compute_confusion = False  # TODO: ATM this is not implemented

batch_size_audio = 1
num_epochs = 50

nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons
LSTM_HIDDEN_LIST = [64,64]
BIDIRECTIONAL = True

CNN_NETWORK = "google"
DENSE_HIDDEN_LIST = [128]

# Decaying LR
LR_start = 0.01
logger_combined.info("LR_start = %s", str(LR_start))
LR_fin = 0.0000001
logger_combined.info("LR_fin = %s", str(LR_fin))
#LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
LR_decay= 0.5
logger_combined.info("LR_decay = %s", str(LR_decay))

#############################################################
# Set locations for DATA, LOG, PARAMETERS, TRAIN info
dataset = "TCDTIMIT"
root_dir = os.path.expanduser('~/TCDTIMIT/combinedSR/' + dataset)
store_dir = root_dir + "/results"
if not os.path.exists(store_dir): os.makedirs(store_dir)

database_binaryDir = root_dir + '/binary'
processedDir = database_binaryDir + "_finalProcessed"
datasetType = "volunteers";


# audio network + cnnNetwork + classifierNetwork
model_name = "RNN__" + str(len(LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join([str(layer) for layer in LSTM_HIDDEN_LIST]) \
             + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if BIDIRECTIONAL else "_unidirectional") +  "_" \
             + "CNN__" + CNN_NETWORK + "_" + '_'.join([str(layer) for layer in DENSE_HIDDEN_LIST]) + "__" + dataset + "_" + datasetType
model_load = os.path.join(store_dir, model_name + ".npz")
model_save = os.path.join(store_dir, model_name)

# for loading stored audio models
audio_model_name = str(len(LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
        [str(layer) for layer in LSTM_HIDDEN_LIST])  + "_nbMFCC" + str(nbMFCCs) + \
        ("_bidirectional" if BIDIRECTIONAL else "_unidirectional") + "_" + dataset
audio_model_dir = os.path.expanduser("~/TCDTIMIT/audioSR/"+dataset+"/results")
audio_model_path = os.path.join(audio_model_dir, audio_model_name + ".npz")

# for loading stored lipreading models
viseme = False; network_type = "google"
lip_model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbPhonemes)
lip_model_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset + "/results"))
lip_model_path = os.path.join(lip_model_dir, lip_model_name+".npz")



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

logger_combined.info('  data source: ' + database_binaryDir)
logger_combined.info('  model target: ' + model_save + '.npz')

storeProcessed = False  # if you have about 10GB hdd space, you can increase the speed by not reprocessing it each iteration
# you can just run this program and it will generate the files the first time it encounters them, or generate them manually with datasetToPkl.py

# just get the names
testVolunteerNumbers = ["13F", "15F", "21M", "23M", "24M", "25M", "28M", "29M", "30F", "31F", "34M", "36F", "37F",
                        "43F", "47M", "51F", "54M"];
testVolunteers = [str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers];
lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
allSpeakers = [f for f in os.listdir(database_binaryDir) if
               os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"]
trainVolunteers = [f for f in allSpeakers if not (f in testVolunteers or f in lipspeakers)];

if datasetType == "combined":
    trainingSpeakerFiles = trainVolunteers + lipspeakers
    testSpeakerFiles = testVolunteers
elif datasetType == "volunteers":
    trainingSpeakerFiles = trainVolunteers
    testSpeakerFiles = testVolunteers
else:
    raise Exception("invalid dataset entered")
datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]


# get a sample of the dataset to debug the network
dataset_test, _, _ = preprocessingCombined.getOneSpeaker(trainingSpeakerFiles[0],
                                                         sourceDataDir=database_binaryDir,
                                                         storeProcessed=True,
                                                         processedDir=processedDir,
                                                         trainFraction=1.0, validFraction=0.0,
                                                         verbose=True)
# import pdb;pdb.set_trace()

##### BUIDING MODEL #####
logger_combined.info('\n* Building network ...')
network = NeuralNetwork('combined', dataset_test,
                            num_features=nbMFCCs, lstm_hidden_list=LSTM_HIDDEN_LIST,
                            num_output_units=nbPhonemes, bidirectional=BIDIRECTIONAL,
                            cnn_network=CNN_NETWORK,
                            dense_hidden_list=DENSE_HIDDEN_LIST,
                            debug=False)

# print number of parameters
nb_params_CNN = lasagne.layers.count_params(network.CNN_lout_features)
nb_params_RNN = lasagne.layers.count_params(network.RNN_lout_features)
nb_params = lasagne.layers.count_params(network.combined_lout)
logger_combined.info(" # params CNN: %s", nb_params_CNN)
logger_combined.info(" # params RNN: %s", nb_params_RNN)
logger_combined.info(" # params combining: %s", nb_params - nb_params_CNN - nb_params_RNN)
logger_combined.info(" # params whole network: %s", nb_params)

# Try to load stored model
logger_combined.info(' Network built. Trying to load stored model: %s', model_load)
success = network.load_model(model_type='combined', model_path=model_load)
if success == -1:
    logger_combined.warning("No complete network found, loading parts...")
    logger_combined.info("CNN : %s", lip_model_path)
    logger_combined.info("RNN : %s", audio_model_path)

    network.load_model(model_type='CNN', model_path=lip_model_path)
    network.load_model(model_type='RNN', model_path=audio_model_path)


##### COMPILING FUNCTIONS #####
logger_combined.info("\n* Compiling functions ...")
network.build_functions(train=True, debug=False)

##### TRAINING #####
logger_combined.info("\n* Training ...")

network.train(datasetFiles, database_binaryDir=database_binaryDir, runType='combined',
                  storeProcessed=True, processedDir=processedDir,
                  num_epochs=num_epochs,
                  batch_size=batch_size_audio, LR_start=LR_start, LR_decay=LR_decay,
                  compute_confusion=False, debug=True, save_name=model_save)

logger_combined.info("\n* Done")
logger_combined.info('Total time: {:.3f}'.format(time.time() - program_start_time))


