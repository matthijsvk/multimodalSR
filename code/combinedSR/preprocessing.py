import timeit;

import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm
import os
from pylearn2.datasets import cache

program_start_time = timeit.default_timer()
import pdb
import python_speech_features

from phoneme_set import phoneme_set_39_list
from general_tools import *


nbPhonemes = 39
phoneme_set_list = phoneme_set_39_list


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