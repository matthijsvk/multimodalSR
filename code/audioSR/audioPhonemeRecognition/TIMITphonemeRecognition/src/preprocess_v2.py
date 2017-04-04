import cPickle
import glob
import math
import os
import sys
import timeit;

import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

program_start_time = timeit.default_timer()
import random
random.seed(int(timeit.default_timer()))
import pdb

import general_tools
import python_speech_features
from phoneme_set import phoneme_set_61, phoneme_set_39, phoneme_set_39_list, phoneme_set_61_list

import logging, colorFormatting  # debug < info < warn < error < critical  # from https://docs.python.org/3/howto/logging-cookbook.html
logging.setLoggerClass(colorFormatting.ColoredLogger)
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

phoneme_set_list = phoneme_set_39_list  #import list of phonemes, convert to dictionary with number mappings (see phoneme_set.py)
values = [i for i in range(0, len(phoneme_set_list))]
phoneme_classes = dict(zip(phoneme_set_list, values))


## DATA LOCATIONS ##
rootPath = "/home/matthijs/TCDTIMIT/TIMIT/fixed" + str(nbPhonemes) + "/TIMIT/"
train_source_path = os.path.join(rootPath, 'TRAIN')
test_source_path = os.path.join(rootPath, 'TEST')

outputDir = "/home/matthijs/TCDTIMIT/TIMIT/binary_list" + str(nbPhonemes) + "/speech2phonemes26Mels/"
target = os.path.join(outputDir, 'std_preprocess_26_ch')
target_path = target + '.pkl'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# already exists, ask if overwrite
if (os.path.exists(target_path)):
    if (not general_tools.query_yes_no(target_path + " exists. Overwrite?", "no")):
        raise Exception("Not Overwriting")


### SETUP ###
if VERBOSE:
    logger.info('VERBOSE mode: \tACTIVE')
else:
    logger.info('VERBOSE mode: \tDEACTIVE')

if DEBUG:
    logger.info('DEBUG mode: \tACTIVE, only a small dataset will be preprocessed')
    target_path = target + '_DEBUG.pkl'
else:
    logger.info('DEBUG mode: \tDEACTIVE')


## Functions ##
def get_total_duration(file):
    """Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
    for line in reversed(list(open(file))):
        [_, val, _] = line.split()
        return int(val)


def create_mfcc(method, filename):
    """Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)

	method is a dummy input!!"""

    (rate, sample) = wav.read(filename)

    mfcc = python_speech_features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                                       preemph=0.97, appendEnergy=True)

    derivative = np.zeros(mfcc.shape)
    for i in range(1, mfcc.shape[0] - 1):
        derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]

    out = np.concatenate((mfcc, derivative), axis=1)

    return out, out.shape[0]


def calc_norm_param(X, VERBOSE=False):
    """Assumes X to be a list of arrays (of differing sizes)"""
    total_len = 0
    mean_val = np.zeros(X[0].shape[1])
    std_val = np.zeros(X[0].shape[1])
    for obs in X:
        obs_len = obs.shape[0]
        mean_val += np.mean(obs, axis=0) * obs_len
        std_val += np.std(obs, axis=0) * obs_len
        total_len += obs_len

    mean_val /= total_len
    std_val /= total_len

    if VERBOSE:
        print(total_len)
        print(mean_val.shape)
        print('  {}'.format(mean_val))
        print(std_val.shape)
        print('  {}'.format(std_val))

    return mean_val, std_val, total_len


def normalize(X, mean_val, std_val):
    for i in range(len(X)):
        X[i] = (X[i] - mean_val) / std_val
    return X


def set_type(X, type):
    for i in range(len(X)):
        X[i] = X[i].astype(type)
    return X


def preprocess_dataset(source_path, verbose=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    X = []
    Y = []

    # source_path should be TRAIN/ or TEST/
    wav_files = sorted(glob.glob(source_path + '/*/*/*.WAV'))
    label_files = sorted(glob.glob(source_path + '/*/*/*.PHN'))
    # import pdb; pdb.set_trace()
    logger.debug("Found %d WAV files" % len(wav_files))
    logger.debug("Found %d PHN files" % len(label_files))
    assert len(wav_files) == len(label_files) != 0

    for i in tqdm(range(len(wav_files))):
        phn_name = str(label_files[i])
        wav_name = str(wav_files[i])

        if (wav_name.startswith("SA")):
            continue

        total_duration = get_total_duration(phn_name)
        fr = open(phn_name)

        X_val, total_frames = create_mfcc('DUMMY', wav_name)
        total_frames = int(total_frames)

        X.append(X_val)

        # some .PHN files don't start at 0. Default phoneme = silence (expected at the end of phoneme_set_list)
        y_val = np.zeros(total_frames) - phoneme_classes[phoneme_set_list[-1]]
        #start_ind = 0
        for line in fr:
            [start_time, end_time, phoneme] = line.rstrip('\n').split()
            start_time = int(start_time)
            start_ind = int(np.round(start_time *  (total_frames / float(total_duration))))
            end_time = int(end_time)
            end_ind = int(np.round(end_time * (total_frames / float(total_duration))))

            phoneme_num = phoneme_classes[phoneme]
            #check that phoneme is found in dict
            if (phoneme_num ==-1):
                logger.debug("In file: %s, phoneme not found: %s", phn_name, phoneme)
                pdb.set_trace()
            y_val[start_ind:end_ind] = phoneme_num

            logger.debug('%s', (total_frames / float(total_duration)))
            logger.debug('TIME  start: %s end: %s, phoneme: %s, class: %s', start_time, end_time, phoneme, phoneme_num)
            logger.debug('FRAME start: %s end: %s, phoneme: %s, class: %s', start_ind, end_ind, phoneme, phoneme_num)
        fr.close()

        if -1 in y_val:
            logger.warning("%s", phn_name)
            logger.warning('WARNING: -1 detected in TARGET: %s', y_val)
            pdb.set_trace()

        Y.append(y_val.astype('int32'))

        if verbose:
            logger.debug('(%s) create_target_vector: %s', i, phn_name[:-4])
            logger.debug('type(X_val): \t\t %s',type(X_val))
            logger.debug('X_val.shape: \t\t %s',X_val.shape)
            logger.debug('type(X_val[0][0]):\t %s',type(X_val[0][0]))

            logger.debug('type(y_val): \t\t %s',type(y_val))
            logger.debug('y_val.shape: \t\t %s',y_val.shape)
            logger.debug('type(y_val[0]):\t %s',type(y_val[0]))
            logger.debug('y_val: \t\t %s',(y_val))


        if DEBUG and i >= debug_size:
            break
    return X, Y


##### PREPROCESSING #####

logger.info('Preprocessing data ...')
logger.info('  Training data: %s ', train_source_path)
X_train_all, y_train_all = preprocess_dataset(train_source_path,
                                              verbose=VERBOSE)
logger.info('  Test data: %s', test_source_path)
X_test, y_test = preprocess_dataset(test_source_path,
                                    verbose=VERBOSE)
# figs = list(map(plt.figure, plt.get_fignums()))

assert len(X_train_all) == len(y_train_all)
assert len(X_test) == len(y_test)
logger.info(' Loading data complete.')

if VERBOSE:
    print('Type and shape/len of X_train_all')
    print('type(X_train_all): {}'.format(type(X_train_all)))
    print('type(X_train_all[0]): {}'.format(type(X_train_all[0])))
    print('type(X_train_all[0][0]): {}'.format(type(X_train_all[0][0])))
    print('type(X_train_all[0][0][0]): {}'.format(type(X_train_all[0][0][0])))


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

if VERBOSE:
    print('Length of train, val, test')
    print("train X: ", len(X_train))
    print("train y: ",len(y_train))

    print("val X: ",len(X_val))
    print("val y: ",len(y_val))

    print("test X: ",len(X_test))
    print("test y: ",len(y_test))


logger.info('Normalizing data ...')
logger.info('    Each channel mean=0, sd=1 ...')

mean_val, std_val, _ = calc_norm_param(X_train)

X_train = normalize(X_train, mean_val, std_val)
X_val = normalize(X_val, mean_val, std_val)
X_test = normalize(X_test, mean_val, std_val)

# make sure we're working with float32
X_data_type = 'float32'
X_train = set_type(X_train, X_data_type)
X_val = set_type(X_val, X_data_type)
X_test = set_type(X_test, X_data_type)

y_data_type = 'int32'
y_train = set_type(y_train, y_data_type)
y_val = set_type(y_val, y_data_type)
y_test = set_type(y_test, y_data_type)

# Convert to numpy arrays
# X_train = np.array(X_train)
# X_val = np.array(X_val)
# X_test = np.array(X_test)
#
# y_train = np.array(y_train)
# y_val = np.array(y_val)
# y_test = np.array(y_test)

if VERBOSE:
    print('X train')
    print(type(X_train), len(X_train))
    print(type(X_train[0]), X_train[0].shape)
    print(type(X_train[0][0]), X_train[0][0].shape)
    print('y train')
    print(type(y_train), len(y_train))
    print(type(y_train[0]), y_train[0].shape)
    print(type(y_train[0][0]), y_train[0][0].shape)

logger.info('Saving data to %s', target_path)
with open(target_path, 'wb') as cPickle_file:
    cPickle.dump(
            [X_train, y_train, X_val, y_val, X_test, y_test],
            cPickle_file,
            protocol=cPickle.HIGHEST_PROTOCOL)

logger.info('Preprocessing complete!')

logger.info('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))
