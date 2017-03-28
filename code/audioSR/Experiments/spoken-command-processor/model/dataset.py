import glob
import itertools
import os

import numpy as np
import tqdm

from . import utils


class TIMITReader(object):
    def __init__(self):
        self.train_dataset_path = os.environ['TIMIT_TRAINING_PATH']
        self.test_dataset_path = os.environ['TIMIT_TESTING_PATH']
        self.data_root = os.environ['MODEL_PARAMETERS']


    def params(self, name, ext='npy'):
        return os.path.join(self.data_dir, name + '.%s' % ext)

    def load_train_data(self, limit=None):
        """
        For self.model == 'speech2phonemes', returns:
            X_train --> [num_of_training_mfcc_vectors, 39]
            y_train --> [num_of_training_mfcc_vectors, 1]
        """
        print('Loading training data...')

        cached = [self.params('X_train'), self.params('y_train')]
        if all(map(os.path.exists, cached)):
            print('Found .npy files for X_train and y_train. Loading...')
            X_train = np.load(self.params('X_train'))
            y_train = np.load(self.params('y_train'))

        else:
            print('Did not find .npy files for X_train and y_train. Parsing dataset...')
            X_train, y_train = self.normalize(
                *self.reader(self.train_dataset_path))

            np.save(self.params('X_train'), X_train)
            np.save(self.params('y_train'), y_train)

        if limit:
            print('Returning %d/%d of the training data...' % (limit, X_train.shape[0]))
            X_train = X_train[:limit, :]
            y_train = y_train[:limit]

        return X_train, y_train

    def load_test_data(self, limit=None):
        """
        For self.model == 'speech2phonemes', returns:
            X_test  --> [num_of_testing_mfcc_vectors, 39]
            y_test  --> [num_of_testing_mfcc_vectors, 1]
        """
        print('Loading testing data...')

        cached = [self.params('X_test'), self.params('y_test')]
        if all(map(os.path.exists, cached)):
            print('Found .npy files for X_test and y_test. Loading...')
            X_test = np.load(self.params('X_test'))
            y_test = np.load(self.params('y_test'))

        else:
            print('Did not find .npy files for X_test and y_test. Parsing dataset...')
            X_test, y_test = self.apply_normalizer(
                *self.reader(self.test_dataset_path))

            np.save(self.params('X_test'), X_test)
            np.save(self.params('y_test'), y_test)

        if limit:
            print('Returning %d/%d of the testing data...' % (limit, X_test.shape[0]))
            X_test = X_test[:limit, :]
            y_test = y_test[:limit]

        return X_test, y_test

    def _parse_timit_line(self, line):
        start_frame, end_frame, label = line.split(' ')

        return int(start_frame), int(end_frame), label.strip('\n')

    def load_unique_phonemes_as_class_numbers(self):
        phonemes = {}

        with open(os.environ['PHONE_LIST_PATH'], 'r') as f:
            class_number = 0

            for ph in map(lambda p: p.strip(), f.readlines()):
                phonemes[ph] = class_number
                class_number += 1

        return phonemes

    def load_unique_words_as_class_numbers(self):
        words = {}

        with open(os.environ['WORD_LIST_PATH'], 'r') as f:
            class_number = 0

            for word in map(lambda w: w.strip(), f.readlines()):
                words[word] = class_number
                class_number += 1

        return words

class Speech2Phonemes(TIMITReader):
    def __init__(self):
        super(Speech2Phonemes, self).__init__()
        print("initialize Speech2Phonemes reader...")

        self.reader = self._read_labeled_wavfiles
        self.data_dir = os.path.join(self.data_root, 'speech2phonemes')
        print("data dir: ", self.data_dir)

        self.normalize = self._wavfile_normalize
        self.apply_normalizer = self._wavfile_apply_normalizer

    def _read_labeled_wavfiles(self, root_timit_path):
        print("Reading from ", root_timit_path)
        wavfiles = sorted(glob.glob(root_timit_path + '/*/*/*.WAV'))
        labels_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))

        print("Found ", len(wavfiles), " WAV files")
        print("Found ", len(labels_files), " PHN files")

        X, y = [], []

        for wf, lf in tqdm(zip(wavfiles, labels_files)):
            for mfccs, label in self._read_labeled_wavfile(wf, lf):
                X.append(mfccs)
                y.append(label)

        # Convert phoneme strings in y_train to class numbers
        phoneme_classes = self.load_unique_phonemes_as_class_numbers()
        y = [phoneme_classes[y[i]] for i in range(len(y))]

        return np.array(X), np.array(y)

    def _read_labeled_wavfile(self, wavfile, labels_file):
        """Map each 20ms recording to a single label."""
        mfccs_and_deltas, segment_duration_frames, hop_duration_frames = utils.wavfile_to_mfccs(wavfile)

        # Pass through the file with the phones
        labels = []

        with open(labels_file, 'r') as f:
            for line in f.readlines():
                start_frame, end_frame, label = self._parse_timit_line(line)

                phn_frames = end_frame - start_frame
                labels.extend([label] * phn_frames)

        classified = []
        curr_frame = curr_mfcc = 0

        while (curr_frame < (len(labels) - segment_duration_frames)):
            label = max(labels[curr_frame:(curr_frame + segment_duration_frames)])

            yield mfccs_and_deltas[:,curr_mfcc], label

            curr_mfcc += 1
            curr_frame += hop_duration_frames

    def _wavfile_normalize(self, X, y):
        print('Normalizing X_train around each MFCC coefficient\'s mean...')
        X = utils.normalize_mean(X, self.params('mfcc_means'))
        return X, y

    def _wavfile_apply_normalizer(self, X, y=None):
        # Use the MFCC means from the training set to normalize X_train
        X = utils.apply_normalize_mean(X, self.params('mfcc_means'))
        return X, y

class Phonemes2Text(TIMITReader):
    def __init__(self):
        super(Phonemes2Text, self).__init__()

        self.reader = self._read_labeled_phnfiles
        self.data_dir = os.path.join(self.data_root, 'phonemes2text')

        self.normalize = self._phnfile_normalize
        self.apply_normalizer = self._phnfile_apply_normalizer

    def _read_labeled_phnfiles(self, root_timit_path):
        phn_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))
        word_files = sorted(glob.glob(root_timit_path + '/*/*/*.WRD'))

        # Each phoneme is mapped to a class number
        phoneme_classes = self.load_unique_phonemes_as_class_numbers()

        # Each word is mapped to a class number
        word_classes = self.load_unique_words_as_class_numbers()

        # Used to get one-hot vectors for each word; this gives its size (4893)
        num_distinct_words = len(word_classes)

        # Max phonemes per word (in the dataset, the largest is "encyclopedias"
        # with 17... we'll go with a few more)
        num_phonemes_per_word = 30

        X, y = [], []

        for pf, wf in zip(phn_files, word_files):
            for word, phonemes_in_word in self._read_labeled_phnfile(pf, wf):
                pclasses = [phoneme_classes[p] for p in phonemes_in_word]

                if pclasses:
                    padded = np.zeros(num_phonemes_per_word)
                    padded[range(len(pclasses))] = pclasses
                    X.append(padded)

                    onehot_word = np.zeros(num_distinct_words)
                    onehot_word[word_classes[word]] = 1
                    y.append(onehot_word)

        # For the training data, these are the shapes
        # (39826 is the number of samples, 30 is the number of phonemes per
        # word and 6102 is the total number of words in the dataset):
        X = np.array(X) # X.shape -> (39826, 30)
        y = np.array(y) # y.shape -> (39826, 6102)
        return X, y

    def _read_labeled_phnfile(self, phn_file, word_file):
        """Map each word to a list of phones (one phone per frame)"""
        phns = []
        with open(phn_file, 'r') as f:
            for line in f.readlines():
                start_frame, end_frame, label = self._parse_timit_line(line)

                phn_frames = end_frame - start_frame
                phns.extend([label] * phn_frames)

        with open(word_file, 'r') as f:
            for line in f.readlines():
                start_frame, end_frame, label = self._parse_timit_line(line)

                with_repeats = phns[start_frame:end_frame]
                word_phns = [k[0] for k in itertools.groupby(with_repeats)]

                yield label, word_phns

    def _phnfile_normalize(self, X, y):
        # No normalization taking place at the moment
        return X, y

    def _phnfile_apply_normalizer(self, X, y):
        # No normalization taking place at the moment
        return X, y
