import cPickle
import glob
import math
import os
import timeit;

import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

program_start_time = timeit.default_timer()
import random

random.seed(int(timeit.default_timer()))
import pdb

import python_speech_features
from audioPhonemeRecognition.phoneme_set import phoneme_set_39_list
import audioPhonemeRecognition.general_tools
import audioPhonemeRecognition.preprocessWavs

import logging, audioPhonemeRecognition.formatting  # debug < info < warn < error < critical
#  from https://docs.python.org/3/howto/logging-cookbook.html

logging.setLoggerClass(audioPhonemeRecognition.formatting.ColoredLogger)
logger = logging.getLogger('PrepData')
logger.setLevel(logging.INFO)

##### SCRIPT META VARIABLES #####
VERBOSE = True
DEBUG = False
debug_size = 50

##### SCRIPT VARIABLES #####

FRAC_TRAIN = 0.9

# TODO MODIFY THESE PARAMETERS for other nbPhonemes. Save location is updated automatically.
nbPhonemes = 39

phoneme_set_list = phoneme_set_39_list  # import list of phonemes, convert to dictionary with number mappings (see phoneme_set.py)
values = [i for i in range(0, len(phoneme_set_list))]
phoneme_classes = dict(zip(phoneme_set_list, values))

## DATA LOCATIONS ##
rootPath = os.path.expanduser("~/TCDTIMIT/audioSR/TIMIT/fixed") + str(nbPhonemes) + "/TIMIT/"
train_source_path = os.path.join(rootPath, 'TRAIN')
test_source_path = os.path.join(rootPath, 'TEST')

outputDir = os.path.expanduser("~/TCDTIMIT/audioSR/TIMIT/binaryOIENOSIRDENRS") + str(nbPhonemes) + "/speech2phonemes26Mels"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

target = os.path.join(outputDir, 'std_preprocess_26_ch')
target_path = target + '.pkl'

# already exists, ask if overwrite
if (os.path.exists(target_path)):
    if (not audioPhonemeRecognition.general_tools.query_yes_no(target_path + " exists. Overwrite?", "no")):
        raise Exception("Not Overwriting")

### SETUP ###
if DEBUG:
    logger.info('DEBUG mode: \tACTIVE, only a small dataset will be preprocessed')
    target_path = target + '_DEBUG.pkl'
else:
    logger.info('DEBUG mode: \tDEACTIVE')
    debug_size=None


##### PREPROCESSING #####

logger.info('Preprocessing data ...')
logger.info('  Training data: %s ', train_source_path)
X_train_all, y_train_all = audioPhonemeRecognition.preprocessWavs.preprocess_dataset(source_path=train_source_path, logger=logger, debug=debug_size)
logger.info('  Test data: %s', test_source_path)
X_test, y_test = audioPhonemeRecognition.preprocessWavs.preprocess_dataset(source_path=test_source_path, logger=logger, debug=debug_size)


assert len(X_train_all) == len(y_train_all)
assert len(X_test) == len(y_test)
logger.info(' Loading data complete.')

logger.debug('Type and shape/len of X_all')
logger.debug('type(X_all): {}'.format(type(X_train_all)))
logger.debug('type(X_all[0]): {}'.format(type(X_train_all[0])))
logger.debug('type(X_all[0][0]): {}'.format(type(X_train_all[0][0])))
logger.debug('type(X_all[0][0][0]): {}'.format(type(X_train_all[0][0][0])))

logger.info('Creating Validation index ...')
test_size = len(X_test)
total_size = len(X_train_all)
train_size = int(math.ceil(total_size * FRAC_TRAIN))

val_size = total_size - train_size
val_idx = random.sample(range(0, total_size), val_size)
val_idx = [int(i) for i in val_idx]

# ensure that the validation set isn't empty
if DEBUG:
    val_idx[0] = 0
    val_idx[1] = 1

logger.info('Separating validation and training set ...')
X_train = []
X_val = []
y_train = []
y_val = []
for i in range(len(X_train_all)):
    if i in val_idx:
        X_val.append(X_train_all[i])
        y_val.append(y_train_all[i])
    else:
        X_train.append(X_train_all[i])
        y_train.append(y_train_all[i])

# Print some information
logger.info('Length of train, val, test')
logger.info("  train X: %s", len(X_train))
logger.info("  train y: %s", len(y_train))

logger.info("  val X: %s", len(X_val))
logger.info("  val y: %s", len(y_val))

logger.info("  test X: %s", len(X_test))
logger.info("  test y: %s", len(y_test))

logger.info('Normalizing data ...')
logger.info('    Each channel mean=0, sd=1 ...')

mean_val, std_val, _ = audioPhonemeRecognition.preprocessWavs.calc_norm_param(X_train)

X_train = audioPhonemeRecognition.preprocessWavs.normalize(X_train, mean_val, std_val)
X_val = audioPhonemeRecognition.preprocessWavs.normalize(X_val, mean_val, std_val)
X_test = audioPhonemeRecognition.preprocessWavs.normalize(X_test, mean_val, std_val)

# make sure we're working with float32
X_data_type = 'float32'
X_train = audioPhonemeRecognition.preprocessWavs.set_type(X_train, X_data_type)
X_val = audioPhonemeRecognition.preprocessWavs.set_type(X_val, X_data_type)
X_test = audioPhonemeRecognition.preprocessWavs.set_type(X_test, X_data_type)

y_data_type = 'int32'
y_train = audioPhonemeRecognition.preprocessWavs.set_type(y_train, y_data_type)
y_val = audioPhonemeRecognition.preprocessWavs.set_type(y_val, y_data_type)
y_test = audioPhonemeRecognition.preprocessWavs.set_type(y_test, y_data_type)

# Convert to numpy arrays
# X_train = np.array(X_train)
# X_val = np.array(X_val)
# X_test = np.array(X_test)
#
# y_train = np.array(y_train)
# y_val = np.array(y_val)
# y_test = np.array(y_test)

logger.debug('X train')
logger.debug('  %s %s', type(X_train), len(X_train))
logger.debug('  %s %s', type(X_train[0]), X_train[0].shape)
logger.debug('  %s %s', type(X_train[0][0]), X_train[0][0].shape)
logger.debug('y train')
logger.debug('  %s %s', type(y_train), len(y_train))
logger.debug('  %s %s', type(y_train[0]), y_train[0].shape)
logger.debug('  %s %s', type(y_train[0][0]), y_train[0][0].shape)


logger.info('Saving data to %s', target_path)
dataList = [X_train, y_train, X_val, y_val, X_test, y_test]
audioPhonemeRecognition.general_tools.saveToPkl(target_path, dataList)

meanStd_path = os.path.dirname(outputDir) + os.sep + os.path.basename(train_source_path) + "MeanStd.pkl"
logger.info('Saving Mean and Std_val to %s', meanStd_path)
dataList = [mean_val, std_val]
audioPhonemeRecognition.general_tools.saveToPkl(meanStd_path, dataList)

logger.info('Preprocessing complete!')

logger.info('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))
