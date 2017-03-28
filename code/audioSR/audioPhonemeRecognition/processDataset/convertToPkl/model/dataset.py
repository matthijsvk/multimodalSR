import glob
import os

import numpy as np
from tqdm import tqdm

import utils


class TIMITReader(object):
    def __init__(self):
        self.train_dataset_path = os.environ['TIMIT_TRAINING_PATH']
        self.test_dataset_path = os.environ['TIMIT_TESTING_PATH']
        self.data_root = os.environ['MODEL_PARAMETERS']

    def params(self, name, ext='npy'):
        return os.path.join(self.data_dir, name + '.%s' % ext)

    def createPKL(self,type):
        import pickle
        print("Generating ", type, " files from dataset...")
        if type == 'all' or type == 'train':
            # X_train, y_train = self.normalize(*self.reader(self.train_dataset_path))
            # output_dim = np.max(y_train) + 1
            # y_train_onehot = utils.onehot_matrix(y_train, output_dim)
            # np.save(self.params('X_train'), X_train)
            # np.save(self.params('y_train'), y_train)
            # np.save(self.params('y_train_onehot'), y_train_onehot)
            #
            # trainData = {'X_train': X_train, 'y_train': y_train,'y_train_onehot':y_train_onehot}
            outputPath = self.params('trainData')
            # output = open(outputPath, 'wb')
            # pickle.dump(trainData, output, 2)
            # output.close()
            print("Train files have been written to: ", outputPath)

        if type == 'all' or type == 'test':
            # X_test, y_test = self.normalize(*self.reader(self.test_dataset_path))
            # output_dim = np.max(y_test) + 1
            # y_test_onehot = utils.onehot_matrix(y_test, output_dim)
            # np.save(self.params('X_test'), X_test)
            # np.save(self.params('y_test'), y_test)
            # np.save(self.params('y_test_onehot'), y_test_onehot)
            #
            # testData = {'X_test': X_test, 'y_test': y_test, 'y_test_onehot': y_test_onehot}
            outputPath = self.params('testData')
            # output = open(outputPath, 'wb')
            # pickle.dump(testData, output, 2)
            # output.close()
            print("Test files have been written to: ", outputPath)

        print("Done.")


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
            y_train_onehot = np.load(self.params('y_train_onehot'))

        else:
            print('Did not find .npy files for X_train and y_train. Parsing dataset to create PKL files...')
            self.createPKL('train')

        if limit:
            print('Returning %d/%d of the training data...' % (limit, X_train.shape[0]))
            X_train = X_train[:limit, :]
            y_train = y_train[:limit]
            y_train_onehot = y_train_onehot[:limit]

        return X_train, y_train, y_train_onehot

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
            print(self.params('X_test'))
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

        for wf, lf in tqdm(zip(wavfiles, labels_files),total = len(wavfiles)):
            for mfccs, label in self._read_labeled_wavfile(wf, lf):
                X.append(mfccs)
                y.append(label)

        # Convert phoneme strings in y_train to class numbers
        from phoneme_set import phoneme_set_39
        phoneme_classes = phoneme_set_39
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
                #print(start_frame,end_frame)

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

