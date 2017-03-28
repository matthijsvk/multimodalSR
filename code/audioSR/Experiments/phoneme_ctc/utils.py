from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import fnmatch

import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from tqdm import tqdm

from phoneme_set import phoneme_set_48, phoneme_48_39, phoneme_set_39

def load_wavPhn(train_path):
    wav_files = []
    phn_files = []
    for dirpath, dirs, files in os.walk(train_path):
        for f in fnmatch.filter(files, '*.wav'):
            wav_files.append(os.path.join(dirpath, f))
            phn_files.append(os.path.join(dirpath, 
                             os.path.splitext(f)[0]+'.phn'))
    return wav_files, phn_files


def load_wav(test_path):
    wav_files = []
    for dirpath, dirs, files in os.walk(test_path):
        for f in fnmatch.filter(files, '*.wav'):
            wav_files.append(os.path.join(dirpath, f))
    return wav_files


def join_features(mfcc, fbank):
    features = np.concatenate((mfcc, fbank), axis=1)
    return features


def process_data(wav_files, phn_files):
    max_step_size = 0
    inputs = []
    targets = []
    for i in tqdm(range(len(wav_files))):
        # extract mfcc features from wav
        (rate, sig) = wav.read(wav_files[i])
        mfcc_feat = mfcc(sig, rate)
        fbank_feat = logfbank(sig, rate)
        acoustic_features = join_features(mfcc_feat, fbank_feat) # time_stamp x n_features

        # extract label from phn
        phn_labels = []
        with open(phn_files[i], 'rb') as csvfile:
            phn_reader = csv.reader(csvfile, delimiter=' ')
            for row in phn_reader:
                if row[2] == 'q':
                    continue
                phn_labels.append(phoneme_set_39[phoneme_48_39.get(row[2], row[2])] - 1)

        inputs.append(acoustic_features)
        targets.append(phn_labels)

    return lists_batches(inputs, targets)


def lists_batches(inputs, targets, batch_size = 16): # the batch size should be same with train.py
    assert len(inputs) == len(targets)

    start, end = (0, batch_size)
    data_batches = []

    n_features = inputs[0].shape[1]
    max_steps = 0
    for inp in inputs:
        max_steps = max(max_steps, inp.shape[0])

    while end <= len(inputs):
        batch_inputs = []
        batch_labels = []
        batch_seq_len = []
        max_batch_seq_len = 0
        for i in range(start, end):
            max_batch_seq_len = max(max_batch_seq_len, inputs[i].shape[0])
            batch_inputs.append(inputs[i]) # [batch_size x time_stamp x num_features ...]
            batch_labels.append(targets[i])
            batch_seq_len.append(inputs[i].shape[0]) # [time_stamp ...]
        batch_pad_inputs = []
        for inp in batch_inputs:
            batch_pad_inputs.append(np.pad(inp, ((0, max_batch_seq_len-inp.shape[0]), (0, 0)), 
                                           mode='constant', constant_values=0))

        data_batches.append((batch_pad_inputs, list_sparse_tensor(batch_labels), batch_seq_len))

        start += batch_size
        end += batch_size

    return data_batches
        

def list_sparse_tensor(batch_labels, num_class=40):
    indices = []
    vals = []
    for batch_id, labels in enumerate(batch_labels):
        for seq_id, label in enumerate(labels):
            indices.append([batch_id, seq_id])
            vals.append(label)
    shape = [len(batch_labels), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices, dtype=np.int64), np.array(vals, dtype=np.int32), np.array(shape, dtype=np.int64))


def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)
    acoustic_features = join_features(mfcc_feat, fbank_feat) # time_stamp x n_features
    return acoustic_features


def process_raw_phn(phn_file):
    phn_labels = []
    with open(phn_file, 'rb') as csvfile:
        phn_reader = csv.reader(csvfile, delimiter=' ')
        for row in phn_reader:
            if row[2] == 'q':
                continue
            phn_labels.append(phoneme_48_39.get(row[2], row[2]))
    return phn_labels
