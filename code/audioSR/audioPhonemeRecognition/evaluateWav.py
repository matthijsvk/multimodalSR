from __future__ import print_function

import copy
import os,warnings
import time
import pdb

program_start_time = time.time()
warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting
logger_evaluate = logging.getLogger('evaluate')
logger_evaluate.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
formatter2 = logging.Formatter('%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_evaluate.addHandler(ch)


print("\n * Importing libraries...")
from RNN_tools_lstm import *
import general_tools
import preprocessWavs
import fixDataset.transform as transform


model_dir = os.path.expanduser("~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/results")
meanStd_path = os.path.expanduser("~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/binary39/TCDTIMITMeanStd.pkl")
store_dir = os.path.expanduser("~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/evaluations")
if not os.path.exists(store_dir): os.makedirs(store_dir)

# get the wav files to evaluate
wavDir = os.path.expanduser('~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/fixed39/TCDTIMIT/volunteers/10M')
name = os.path.basename(os.path.dirname(wavDir)) + "_" + os.path.basename(wavDir)


# From WAVS, generate X, y and masks; also store as store_dir/name.pkl
def preprocessLabeledWavs(wavDir, store_dir, name):
    # fixWavs -> suppose this is done
    # convert to pkl
    X, y = preprocessWavs.preprocess_dataset(source_path=wavDir, logger=logger_evaluate)

    X_data_type = 'float32'
    X = preprocessWavs.set_type(X, X_data_type)
    y_data_type = 'int32'
    y = preprocessWavs.set_type(y, y_data_type)

    general_tools.saveToPkl(store_dir + os.sep + name + '.pkl', [X, y])

    return X, y

def preprocessUnlabeledWavs(wavDir, store_dir, name):
    # fixWavs -> suppose this is done
    # convert to pkl
    X = preprocessWavs.preprocess_unlabeled_dataset(source_path=wavDir, logger=logger_evaluate)

    X_data_type = 'float32'
    X = preprocessWavs.set_type(X, X_data_type)

    general_tools.saveToPkl(store_dir + os.sep + name + '.pkl', [X, masks])

    return X

wav_files = transform.loadWavs(wavDir)
label_files = transform.loadPhns(wavDir)

# if source dir doesn't contain labelfiles, we can't calculate accuracy
calculateAccuracy = True
if not (len(wav_files) == len(label_files)):
    calculateAccuracy = False
    inputs = preprocessUnlabeledWavs(wavDir=wavDir, store_dir=store_dir, name=name)
else: inputs, targets = preprocessLabeledWavs(wavDir=wavDir, store_dir=store_dir, name=name)


# normalize inputs, convert to float32
with open(meanStd_path, 'rb') as cPickle_file:
    [mean_val, std_val] = cPickle.load(cPickle_file)
inputs = preprocessWavs.normalize(inputs, mean_val, std_val)

# just to be sure
X_data_type = 'float32'
inputs = preprocessWavs.set_type(inputs, X_data_type)


# Print some information
logger_evaluate.info("\n* Data information")
logger_evaluate.info('  inputs')
logger_evaluate.info('%s %s', type(inputs), len(inputs))
logger_evaluate.info('%s %s', type(inputs[0]), inputs[0].shape)
logger_evaluate.info('%s %s', type(inputs[0][0]), inputs[0][0].shape)
logger_evaluate.info('%s', type(inputs[0][0][0]))
logger_evaluate.info('y train')
logger_evaluate.info('  %s %s', type(targets), len(targets))
logger_evaluate.info('  %s %s', type(targets[0]), targets[0].shape)
logger_evaluate.info('  %s %s', type(targets[0][0]), targets[0][0].shape)


# log file
logFile = store_dir + os.sep + name + '.log'
fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_evaluate.addHandler(fh)


##### SCRIPT META VARIABLES #####
INPUT_SIZE = 26  # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
NUM_OUTPUT_UNITS = 39
N_HIDDEN = 400
N_HIDDEN_2 = 0
BIDIRECTIONAL = True
MOMENTUM = 0.9

#############################################################
# Set locations for LOG, PARAMETERS, TRAIN info
model_name = "1HiddenLayer" + str(N_HIDDEN) + "_nbMFCC" + str(INPUT_SIZE) + (
"_bidirectional" if BIDIRECTIONAL else "_unidirectional") + ".npz"

# model parameters and network_training_info
model_load = os.path.join(model_dir, model_name)
#############################################################

##### BUIDING MODEL #####
logger_evaluate.info('\n* Building network ...')
RNN_network = NeuralNetwork('RNN', batch_size=1, num_features=INPUT_SIZE, n_hidden=N_HIDDEN,
                            num_output_units=NUM_OUTPUT_UNITS, bidirectional=BIDIRECTIONAL, seed=0, debug=False, logger=logger_evaluate)

# Try to load stored model
logger_evaluate.info(' Network built. Trying to load stored model: %s', model_load)
RNN_network.load_model(model_load, logger=logger_evaluate)

##### COMPILING FUNCTIONS #####
logger_evaluate.info("\n* Compiling functions ...")
RNN_network.build_functions(debug=True, train=False, logger=logger_evaluate)


##### EVALUATION #####
logger_evaluate.info("\n* Evaluating: pass over Evaluation Set")
validate_fn = RNN_network.validate_fn
predictions_fn = RNN_network.predictions_fn

# make copy because we might need to use is again for calculating accurasy, and the iterator will remove elements from the array
inputs_bak = copy.deepcopy(inputs)
targets_bak = copy.deepcopy(targets)

# calculate network predictions
predictions = []
for inputs, masks, seq_lengths in iterate_minibatches_noTargets(inputs, batch_size=1, shuffle=False):
    nb_inputs = len(inputs) #usually batch size, but could be lower
    seq_len = len(inputs[0])
    prediction = predictions_fn(inputs, masks)
    prediction = np.reshape(prediction,(nb_inputs,-1))
    prediction = list(prediction)
    predictions= predictions + prediction
    #pdb.set_trace()
predictions_path = store_dir + os.sep + name + "_predictions.pkl"
logger_evaluate.info(" Predictions are being stored under: %s", predictions_path)

# restore inputs
inputs = inputs_bak

# if .phn files are provided, we can check our predictions
totError = 0
totAcc = 0
n_batches = 0
if calculateAccuracy:
    logger_evaluate.info("also calculating accuracy...")
    for inputs, targets, masks, seq_lengths in iterate_minibatches(inputs, targets, batch_size=1, shuffle=False):
        # print(inputs)
        # print(targets)
        # print(masks)
        # print(seq_lengths)
        error, accuracy = validate_fn(inputs, masks, targets)
        totError += error
        totAcc += accuracy
        n_batches += 1

    avg_error = (100 - totError / n_batches * 100)
    avg_Acc = (100 - totAcc / n_batches * 100)

    logger_evaluate.info(" Accuracy: %s", avg_Acc)

    general_tools.saveToPkl(predictions_path, [inputs_bak, predictions, targets_bak, avg_Acc])
else:
    inputs = general_tools.pad_sequences_X(inputs_bak)
    general_tools.saveToPkl(predictions_path, [inputs_bak, predictions])


logger_evaluate.info("\n* Done")
logger_evaluate.info('Total time: {:.3f}'.format(time.time() - program_start_time))
