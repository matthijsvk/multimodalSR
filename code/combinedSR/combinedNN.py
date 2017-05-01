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

#############################################################

##### SCRIPT META VARIABLES #####
VERBOSE = True
compute_confusion = False  # TODO: ATM this is not implemented

batch_size_audio = 1
num_epochs = 20

nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons
LSTM_HIDDEN_LIST = [256,256]
BIDIRECTIONAL = True

# lipreading
CNN_NETWORK = "google"
# using CNN-LSTM combo: what to input to LSTM? direct conv outputs or first through dense layers?
cnn_features = 'conv' #'dense' # 39 outputs as input to LSTM
LIP_RNN_HIDDEN_LIST = None#[256,256]  # set to None to disable CNN-LSTM architecture

# after concatenation of audio and lipreading, which dense layers before softmax?
DENSE_HIDDEN_LIST = [2048,2048,512] #[128,128,128,128]

# Decaying LR
LR_start = 0.001
logger_combined.info("LR_start = %s", str(LR_start))
LR_fin = 0.0000001
logger_combined.info("LR_fin = %s", str(LR_fin))
#LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
LR_decay= 0.7071
logger_combined.info("LR_decay = %s", str(LR_decay))

# Set locations for DATA, LOG, PARAMETERS, TRAIN info
dataset = "TCDTIMIT"
root_dir = os.path.expanduser('~/TCDTIMIT/combinedSR/' + dataset)
store_dir = root_dir + "/results"
if not os.path.exists(store_dir): os.makedirs(store_dir)

database_binaryDir = root_dir + '/binary'
processedDir = database_binaryDir + "_finalProcessed"
datasetType = "lipspeakers" #""volunteers";

# which part of the network to train/save/...
# runType = 'audio'
# runType = 'lipreading'
runType = 'combined'
###########################


# audio network + cnnNetwork + classifierNetwork
model_name = "RNN__" + str(len(LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join([str(layer) for layer in LSTM_HIDDEN_LIST]) \
                     + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if BIDIRECTIONAL else "_unidirectional") +  "__" \
             + "CNN_" + CNN_NETWORK + "_" + cnn_features \
             + ("_lipRNN_" if LIP_RNN_HIDDEN_LIST != None else "") + ('_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST]) if LIP_RNN_HIDDEN_LIST != None else "")  + "__" \
             + "FC_" + '_'.join([str(layer) for layer in DENSE_HIDDEN_LIST]) + "__" \
             + dataset + "_" + datasetType
model_load = os.path.join(store_dir, model_name + ".npz")
model_save = os.path.join(store_dir, model_name)

# for loading stored audio models
audio_dataset = "combined" # TCDTIMIT + TIMIT datasets
audio_model_name = str(len(LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
        [str(layer) for layer in LSTM_HIDDEN_LIST])  + "_nbMFCC" + str(nbMFCCs) + \
        ("_bidirectional" if BIDIRECTIONAL else "_unidirectional") + "_" + audio_dataset
audio_model_dir = os.path.expanduser("~/TCDTIMIT/audioSR/"+audio_dataset+"/results")
audio_model_path = os.path.join(audio_model_dir, audio_model_name + ".npz")

# for loading stored lipreading models
lip_model_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset + "/results"))
viseme = False; network_type = "google"
lip_CNN_model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbPhonemes)
CNN_model_path = os.path.join(lip_model_dir, lip_CNN_model_name + ".npz")

# for CNN-LSTM networks
if LIP_RNN_HIDDEN_LIST != None:
    lip_CNN_LSTM_model_name = lip_CNN_model_name + "_LSTM" + '_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST])
    lip_CNN_LSTM_model_path = os.path.join(lip_model_dir, lip_CNN_LSTM_model_name + ".npz")

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
else:# datasetType == "volunteers":
    trainingSpeakerFiles = trainVolunteers
    testSpeakerFiles = testVolunteers
# else:
#     raise Exception("invalid dataset entered")
datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]


# get a sample of the dataset to debug the network
dataset_test, _, _ = preprocessingCombined.getOneSpeaker(trainingSpeakerFiles[0],
                                                         sourceDataDir=database_binaryDir,
                                                         storeProcessed=True,
                                                         processedDir=processedDir,
                                                         trainFraction=1.0, validFraction=0.0,
                                                         verbose=False)
# import pdb;pdb.set_trace()

## TEST: only lipspeakers
lipspkr_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTrain.pkl")
data = unpickle(lipspkr_path)

##### BUIDING MODEL #####
logger_combined.info('\n\n* Building network ...')
network = NeuralNetwork('combined', dataset=data, loadPerSpeaker = False, #dataset_test,
                            num_features=nbMFCCs, lstm_hidden_list=LSTM_HIDDEN_LIST,
                            num_output_units=nbPhonemes, bidirectional=BIDIRECTIONAL,
                            cnn_network=CNN_NETWORK, cnn_features = cnn_features,
                            lipRNN_hidden_list=LIP_RNN_HIDDEN_LIST,
                            dense_hidden_list=DENSE_HIDDEN_LIST,
                            debug=False)

# print number of parameters
nb_params_CNN_noDense   = lasagne.layers.count_params(network.CNN_lout_features)
nb_params_CNN           = lasagne.layers.count_params(network.CNN_lout)
nb_params_lipreading    = lasagne.layers.count_params(network.lipreading_lout_features)
nb_params_RNN           = lasagne.layers.count_params(network.audioNet_lout_features)
nb_params               = lasagne.layers.count_params(network.combined_lout)
logger_combined.info(" # params lipreading Total: %s", nb_params_lipreading)

if LIP_RNN_HIDDEN_LIST != None:
    logger_combined.info(" # params lipRNN:           %s", nb_params_lipreading - nb_params_CNN)

if cnn_features == 'dense':
    logger_combined.info(" # params CNN:              %s", nb_params_CNN)
else:
    logger_combined.info(" # params CNN:              %s", nb_params_CNN_noDense)

logger_combined.info(" # params audio LSTM:       %s", nb_params_RNN)
logger_combined.info(" # params combining FC:     %s", nb_params - nb_params_lipreading - nb_params_RNN)
logger_combined.info(" # params whole network:    %s", nb_params)



# Try to load stored model
logger_combined.info(' Network built. \n\nTrying to load stored model: %s', model_load)
success = network.load_model(model_type='combined', model_path=model_load)
if success == -1:
    logger_combined.warning("No complete network found, loading parts...")

    logger_combined.info("CNN : %s", CNN_model_path)
    network.load_model(model_type='CNN', model_path=CNN_model_path)

    if LIP_RNN_HIDDEN_LIST != None:
        logger_combined.info("CNN_LSTM : %s", lip_CNN_LSTM_model_path)
        network.load_model(model_type='CNN_LSTM', model_path=lip_CNN_LSTM_model_path)

    logger_combined.info("RNN : %s", audio_model_path)
    network.load_model(model_type='RNN', model_path=audio_model_path)


##### COMPILING FUNCTIONS #####
logger_combined.info("\n\n* Compiling functions ...")
network.build_functions(train=True, debug=False)

##### TRAINING #####
logger_combined.info("\n\n* Training ...")

if runType == 'audio':                  model_save = audio_model_path
elif runType == 'lipreading':
    if LIP_RNN_HIDDEN_LIST != None:     model_save = lip_CNN_LSTM_model_path
    else:                               model_save = CNN_model_path
elif runType == 'combined':             model_save = model_load
else: raise IOError("can't save network params; network output not found")
model_save = model_save.replace(".npz","")

# ### test ###
model_save = model_save + "__test"
###

network.train(datasetFiles, database_binaryDir=database_binaryDir, runType=runType,
                  storeProcessed=True, processedDir=processedDir,
                  num_epochs=num_epochs,
                  batch_size=batch_size_audio, LR_start=LR_start, LR_decay=LR_decay,
                  compute_confusion=False, debug=False, save_name=model_save)

logger_combined.info("\n\n* Done")
logger_combined.info('Total time: {:.3f}'.format(time.time() - program_start_time))


