import cPickle
import glob
import math
import os
import timeit;
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm
program_start_time = timeit.default_timer()
import pdb
import python_speech_features

from phoneme_set import phoneme_set_39_list
import general_tools
import fixDataset.transform as transform


nbPhonemes = 39
phoneme_set_list = phoneme_set_39_list  # import list of phonemes,
# convert to dictionary with number mappings (see phoneme_set.py)
values = [i for i in range(0, len(phoneme_set_list))]
phoneme_classes = dict(zip(phoneme_set_list, values))


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


def calc_norm_param(X):
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

    return mean_val, std_val, total_len


def normalize(X, mean_val, std_val):
    for i in range(len(X)):
        X[i] = (X[i] - mean_val) / std_val
    return X


def set_type(X, type):
    for i in range(len(X)):
        X[i] = X[i].astype(type)
    return X


def preprocess_dataset(source_path, logger=None, debug=None, verbose=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    X = []
    Y = []

    # source_path should be TRAIN/ or TEST/
    wav_files = transform.loadWavs(source_path)
    label_files = transform.loadPhns(source_path)

    # wav_files = sorted(glob.glob(source_path + '/*/*/*.WAV'))
    # label_files = sorted(glob.glob(source_path + '/*/*/*.PHN'))

    # import pdb; pdb.set_trace()
    logger.debug("Found %d WAV files" % len(wav_files))
    logger.debug("Found %d PHN files" % len(label_files))
    assert len(wav_files) == len(label_files)
    assert len(wav_files) != 0

    processed = 0
    for i in tqdm(range(len(wav_files))):
        phn_name = str(label_files[i])
        wav_name = str(wav_files[i])

        if (wav_name.startswith("SA")):  #specific for TIMIT: these files contain strong dialects; don't use
            continue

        total_duration = get_total_duration(phn_name)
        fr = open(phn_name)

        X_val, total_frames = create_mfcc('DUMMY', wav_name)
        total_frames = int(total_frames)

        X.append(X_val)

        # some .PHN files don't start at 0. Default phoneme = silence (expected at the end of phoneme_set_list)
        y_val = np.zeros(total_frames) - phoneme_classes[phoneme_set_list[-1]]
        # start_ind = 0
        for line in fr:
            [start_time, end_time, phoneme] = line.rstrip('\n').split()
            start_time = int(start_time)
            start_ind = int(np.round(start_time * (total_frames / float(total_duration))))
            end_time = int(end_time)
            end_ind = int(np.round(end_time * (total_frames / float(total_duration))))

            phoneme_num = phoneme_classes[phoneme]
            # check that phoneme is found in dict
            if (phoneme_num == -1):
                logger.error("In file: %s, phoneme not found: %s", phn_name, phoneme)
                pdb.set_trace()
            y_val[start_ind:end_ind] = phoneme_num

            if verbose:
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
            logger.debug('type(X_val): \t\t %s', type(X_val))
            logger.debug('X_val.shape: \t\t %s', X_val.shape)
            logger.debug('type(X_val[0][0]):\t %s', type(X_val[0][0]))

            logger.debug('type(y_val): \t\t %s', type(y_val))
            logger.debug('y_val.shape: \t\t %s', y_val.shape)
            logger.debug('type(y_val[0]):\t %s', type(y_val[0]))
            logger.debug('y_val: \t\t %s', (y_val))

        processed+=1
        if debug!=None and processed >= debug:
            break

    return X, Y


def preprocess_unlabeled_dataset(source_path, verbose=False, logger=None):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    X = []

    # source_path should be TRAIN/ or TEST/
    wav_files = transform.loadWavs(source_path)

    # import pdb; pdb.set_trace()
    logger.debug("Found %d WAV files" % len(wav_files))
    assert len(wav_files) != 0

    for i in tqdm(range(len(wav_files))):
        wav_name = str(wav_files[i])
        X_val, total_frames = create_mfcc('DUMMY', wav_name)
        X.append(X_val)

        if verbose:
            logger.debug('type(X_val): \t\t %s', type(X_val))
            logger.debug('X_val.shape: \t\t %s', X_val.shape)
            logger.debug('type(X_val[0][0]):\t %s', type(X_val[0][0]))
    return X