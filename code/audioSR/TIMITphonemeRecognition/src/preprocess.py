import os
import timeit;
import wave

program_start_time = timeit.default_timer()
import random;

random.seed(int(timeit.default_timer()))

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from general_tools import *
import python_speech_features
from tqdm import tqdm

# https://github.com/jameslyons/python_speech_features


##### SCRIPT META VARIABLES #####
VERBOSE = False
DEBUG = True
debug_size = 5
# Convert only a reduced dataset
visualize = False

##### SCRIPT VARIABLES #####
train_size = 3696
val_size = 184
test_size = 1344

data_type = 'float32'

rootPath = "/home/matthijs/TCDTIMIT/TIMIT/fixedWav/TIMIT/"
train_source_path = os.path.join(rootPath, 'TRAIN')
test_source_path = os.path.join(rootPath, 'TEST')
target_path = os.path.join(rootPath, 'std_preprocess_26_ch')

##### SETUP #####
if VERBOSE:
    print('VERBOSE mode: \tACTIVE')
else:
    print('VERBOSE mode: \tDEACTIVE')

if DEBUG:
    print('DEBUG mode: \tACTIVE, only a small dataset will be preprocessed')
    target_path += '_DEBUG'
else:
    print('DEBUG mode: \tDEACTIVE')

phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh",
            "z", "zh",
            "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
            "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
            "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]


# 61 different phonemes

def get_total_duration(file):
    """Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
    for line in reversed(list(open(file))):
        [_, val, _] = line.split()
        return int(val)


def find_phoneme(phoneme_idx):
    for i in range(len(phonemes)):
        if phoneme_idx == phonemes[i]:
            return i
    print("PHONEME NOT FOUND, NaN CREATED!")
    print("\t" + phoneme_idx + " wasn't found!")
    return -1


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


def preprocess_dataset(source_path, VERBOSE=False, visualize=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    i = 0
    X = []
    Y = []

    for dirName, subdirList, fileList in os.walk(source_path):
        for fname in tqdm(fileList,total=len(fileList)):
            if not fname.endswith('.PHN') or (fname.startswith("SA")):
                continue

            phn_fname = dirName + '/' + fname
            wav_fname = dirName + '/' + fname[0:-4] + '.WAV'

            total_duration = get_total_duration(phn_fname)
            fr = open(phn_fname)

            X_val, total_frames = create_mfcc('DUMMY', wav_fname)
            total_frames = int(total_frames)

            X.append(X_val)

            y_val = np.zeros(total_frames) - 1
            start_ind = 0
            for line in fr:
                [start_time, end_time, phoneme] = line.rstrip('\n').split()
                start_time = int(start_time)
                end_time = int(end_time)

                phoneme_num = find_phoneme(phoneme)
                end_ind = np.round((end_time) / total_duration * total_frames)
                y_val[start_ind:end_ind] = phoneme_num

                start_ind = end_ind
            fr.close()

            if -1 in y_val:
                print('WARNING: -1 detected in TARGET')
                print(y_val)

            Y.append(y_val.astype('int32'))

            i += 1
            if VERBOSE:
                print("")
                print('({}) create_target_vector: {}'.format(i, phn_fname[:-4]))
                print('type(X_val): \t\t {}'.format(type(X_val)))
                print('X_val.shape: \t\t {}'.format(X_val.shape))
                print('type(X_val[0][0]):\t {}'.format(type(X_val[0][0])))

        if i >= debug_size and DEBUG:
            break
    print("")
    return X, Y


##### PREPROCESSING #####
print("")
print('Creating Validation index ...')
val_idx = random.sample(range(0, train_size), val_size)
val_idx = [int(i) for i in val_idx]
# ensure that the validation set isn't empty
if DEBUG:
    val_idx[0] = 0
    val_idx[1] = 1

print('Preprocessing data ...')
print('  This will take a while')
X_train_all, y_train_all = preprocess_dataset(train_source_path,
                                                 VERBOSE=False, visualize=False)
X_test, y_test = preprocess_dataset(test_source_path,
                                               VERBOSE=False, visualize=visualize)
# figs = list(map(plt.figure, plt.get_fignums()))

print(' Loading data complete.')

if VERBOSE:
    print("")
    print('Type and shape/len of X_train_all')
    print('type(X_train_all): {}'.format(type(X_train_all)))
    print('type(X_train_all[0]): {}'.format(type(X_train_all[0])))
    print('type(X_train_all[0][0]): {}'.format(type(X_train_all[0][0])))
    print('type(X_train_all[0][0][0]): {}'.format(type(X_train_all[0][0][0])))

print('Separating validation and training set ...')
X_train = [];
X_val = []
y_train = [];
y_val = []
for i in range(len(X_train_all)):
    if i in val_idx:
        X_val.append(X_train_all[i])
        y_val.append(y_train_all[i])
    else:
        X_train.append(X_train_all[i])
        y_train.append(y_train_all[i])

if VERBOSE:
    print("")
    print('Length of train, val, test')
    print(len(X_train))
    print(len(y_train))

    print(len(X_val))
    print(len(y_val))

    print(len(X_test))
    print(len(y_test))

if VERBOSE:
    print("")
    print('Type of train')
    print(type(X_train))
    print(type(y_train))
    print(type(X_train[0]), X_train[0].shape)
    print(type(y_train[0]), y_train[0].shape)

print("")
print('Normalizing data ...')
print('    Each channel mean=0, sd=1 ...')

mean_val, std_val, _ = calc_norm_param(X_train)

X_train = normalize(X_train, mean_val, std_val)
X_val = normalize(X_val, mean_val, std_val)
X_test = normalize(X_test, mean_val, std_val)

X_train = set_type(X_train, data_type)
X_val = set_type(X_val, data_type)
X_test = set_type(X_test, data_type)

if visualize == True:
    for i in range(debug_size):
        plt.figure(i)
        plt.subplot(4, 1, 4)

        plt.imshow(X_test[i].T, interpolation='nearest', aspect='auto')
        # plt.axis('off')
        # plt.title('Preprocessed data')

        plt.ylabel('Normalized data')
        plt.tick_params(
                axis='both',  # changes apply to the axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom
                top='off',  # ticks along the top
                right='off',  # ticks along the right
                left='off',  # ticks along the left
                labelbottom='off',  # labels along the bottom
                labelleft='off')  # labels along the top

    plt.show()

print('Saving data ...')
storePath = target_path + '.pkl'
with open(target_path + '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
            [X_train, y_train, X_val, y_val, X_test, y_test],
            cPickle_file,
            protocol=cPickle.HIGHEST_PROTOCOL)

print('Preprocessing complete!')
print("")

print('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))



