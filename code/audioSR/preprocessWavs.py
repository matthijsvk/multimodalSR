import timeit;

import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

program_start_time = timeit.default_timer()
import pdb
import python_speech_features

from phoneme_set import phoneme_set_39_list
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


def create_mfcc(method, filename, type=2):
    """Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)

	method is a dummy input!!"""

    (rate, sample) = wav.read(filename)

    mfcc = python_speech_features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                                       preemph=0.97, appendEnergy=True)
    out = mfcc
    if type > 13:
        derivative = np.zeros(mfcc.shape)
        for i in range(1, mfcc.shape[0] - 1):
            derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]

        mfcc_derivative = np.concatenate((mfcc, derivative), axis=1)
        out = mfcc_derivative
        if type > 26:
            derivative2 = np.zeros(derivative.shape)
            for i in range(1, derivative.shape[0] - 1):
                derivative2[i, :] = derivative[i + 1, :] - derivative[i - 1, :]

            out = np.concatenate((mfcc, derivative, derivative2), axis=1)
            if type > 39:
                derivative3 = np.zeros(derivative2.shape)
                for i in range(1, derivative2.shape[0] - 1):
                    derivative3[i, :] = derivative2[i + 1, :] - derivative2[i - 1, :]

                out = np.concatenate((mfcc, derivative, derivative2, derivative3), axis=1)

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


def preprocess_dataset(source_path, nbMFCCs=39, logger=None, debug=None, verbose=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    X = []
    y = []
    valid_frames = []

    print(nbMFCCs)

    # source_path is the root dir of all the wav/phn files
    wav_files = transform.loadWavs(source_path)
    label_files = transform.loadPhns(source_path)

    logger.debug("Found %d WAV files" % len(wav_files))
    logger.debug("Found %d PHN files" % len(label_files))
    assert len(wav_files) == len(label_files)
    assert len(wav_files) != 0

    processed = 0
    for i in tqdm(range(len(wav_files))):
        phn_name = str(label_files[i])
        wav_name = str(wav_files[i])

        if (wav_name.startswith("SA")):  #specific for TIMIT: these files contain strong dialects; don't use them
            continue

        # Get MFCC of the WAV
        X_val, total_frames = create_mfcc('DUMMY', wav_name, nbMFCCs)  # get 3 levels: 0th, 1st and 2nd derivative (=> 3*13 = 39 coefficients)
        total_frames = int(total_frames)

        X.append(X_val)


        # Get phonemes and valid frame numbers out of .phn files
        total_duration = get_total_duration(phn_name)
        fr = open(phn_name)

        # some .PHN files don't start at 0. Set default phoneme to silence (expected at the end of phoneme_set_list)
        y_vals = np.zeros(total_frames) + phoneme_classes[phoneme_set_list[-1]]
        valid_frames_vals = []

        for line in fr:
            [start_time, end_time, phoneme] = line.rstrip('\n').split()
            start_time = int(start_time)
            start_ind = int(np.round(start_time * (total_frames / float(total_duration))))
            end_time = int(end_time)
            end_ind = int(np.round(end_time * (total_frames / float(total_duration))))

            valid_ind = int( (start_ind + end_ind)/2)
            valid_frames_vals.append(valid_ind)

            phoneme_num = phoneme_classes[phoneme]
            # check that phoneme is found in dict
            if (phoneme_num == -1):
                logger.error("In file: %s, phoneme not found: %s", phn_name, phoneme)
                pdb.set_trace()
            y_vals[start_ind:end_ind] = phoneme_num

            if verbose:
                logger.debug('%s', (total_frames / float(total_duration)))
                logger.debug('TIME  start: %s end: %s, phoneme: %s, class: %s', start_time, end_time, phoneme, phoneme_num)
                logger.debug('FRAME start: %s end: %s, phoneme: %s, class: %s', start_ind, end_ind, phoneme, phoneme_num)
        fr.close()

        # append the target array to our y
        y.append(y_vals.astype('int32'))

        # append the valid_frames array to our valid_frames
        valid_frames_vals = np.array(valid_frames_vals)
        valid_frames.append(valid_frames_vals.astype('int32'))


        if verbose:
            logger.debug('(%s) create_target_vector: %s', i, phn_name[:-4])
            logger.debug('type(X_val): \t\t %s', type(X_val))
            logger.debug('X_val.shape: \t\t %s', X_val.shape)
            logger.debug('type(X_val[0][0]):\t %s', type(X_val[0][0]))

            logger.debug('type(y_val): \t\t %s', type(y_vals))
            logger.debug('y_val.shape: \t\t %s', y_vals.shape)
            logger.debug('type(y_val[0]):\t %s', type(y_vals[0]))
            logger.debug('y_val: \t\t %s', (y_vals))

        processed += 1
        if debug != None and processed >= debug:
            break

    return X, y, valid_frames


def preprocess_unlabeled_dataset(source_path, verbose=False, logger=None): # TODO
    wav_files = transform.loadWavs(source_path)
    logger.debug("Found %d WAV files" % len(wav_files))
    assert len(wav_files) != 0

    X = []
    for i in tqdm(range(len(wav_files))):
        wav_name = str(wav_files[i])
        X_val, total_frames = create_mfcc('DUMMY', wav_name)
        X.append(X_val)

        if verbose:
            logger.debug('type(X_val): \t\t %s', type(X_val))
            logger.debug('X_val.shape: \t\t %s', X_val.shape)
            logger.debug('type(X_val[0][0]):\t %s', type(X_val[0][0]))
    return X