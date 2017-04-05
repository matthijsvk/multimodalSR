import glob
import logging
import os
import sys

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

import utils

"""
This file is based on https://github.com/dtjchen/spoken-command-processor
"""

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
logging.debug('A debug message!')

# where are the source files?
train_dataset_path = "/home/matthijs/TCDTIMIT/TIMIT/fixed/TIMIT/TRAIN"
test_dataset_path = "/home/matthijs/TCDTIMIT/TIMIT/fixed/TIMIT/TEST"

# where to store output?
data_root = "/home/matthijs/TCDTIMIT/TIMIT/binary"
data_dir = os.path.join(data_root, 'speech2phonemes26Mels')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
print("data dir: ", data_dir)


#########################
dataset_to_create = 'all'
#########################


def createPKL(type):
    import pickle
    print("Generating ", type, " files from dataset...")

    if type == 'all':
        createPKL('train')
        createPKL('test')
        return 0
    elif type == 'train':
        outputPath = getFileName(data_dir, 'trainData', 'pkl')
        print(outputPath)
        if (os.path.exists(outputPath)):
            if (not utils.query_yes_no("TrainData.pkl exists. Overwrite?", "no")):
                return 0
        X, y = read_labeled_wavfiles(train_dataset_path)
        print("ALL FILES READ")
        print(X.shape, y.shape)
        mean_val = getMean(X)
        np.save(getFileName(data_dir, 'Train_mean'), np.array([mean_val]))
        X = apply_normalize(X, mean_val)
    elif type == 'test':
        outputPath = getFileName(data_dir, 'testData' ,'pkl')
        if (os.path.exists(outputPath)):
            if (not utils.query_yes_no("TestData.pkl exists. Overwrite?", "no")):
                return 0
        X, y = read_labeled_wavfiles(test_dataset_path)

        # we need to use mean and std from the train set
        try: mean_val = np.load(getFileName(data_dir, 'Train_mean'))
        except: "you have to generate the train dataset first"
        X = apply_normalize(X, mean_val)
    else:
        raise ValueError("createPKL: COMMAND NOT RECOGNIZED")

    output_dim = np.max(y) + 1
    y_onehot = utils.onehot_matrix(y, output_dim)
    np.save(getFileName(data_dir,'X'+type), X)
    np.save(getFileName(data_dir,'y'+type), y)
    np.save(getFileName(data_dir,'y_'+type+'onehot'), y_onehot)

    data = {'X_'+type: X, 'y_'+type: y, 'y_'+type+'_onehot': y_onehot}
    output = open(outputPath, 'wb')
    pickle.dump(data, output, 2)
    output.close()
    print(type+" files have been written to: ", outputPath)

    print("Done.")
    return 0

def load_data(type, limit=None):
    """returns:
        X_ --> [num_of_ing_mfcc_vectors, 39]
        y_ --> [num_of_ing_mfcc_vectors, 1]
    """
    print('Loading data...')

    cached = [getFileName(data_dir,'X'), getFileName(data_dir,'y')]
    if all(map(os.path.exists, cached)):
        print('Found .npy files for X and y. Loading...')
        X = np.load(getFileName(data_dir,'X'))
        y = np.load(getFileName(data_dir,'y'))
        y_onehot = np.load(getFileName(data_dir,'y_onehot'))

    else:
        print('Did not find .npy files for '+type+' X and y. Parsing dataset to create PKL files...')
        createPKL(type)

    if limit:
        print('Returning %d/%d of the data...' % (limit, X.shape[0]))
        X_ = X[:limit, :]
        y = y[:limit]
        y_onehot = y_onehot[:limit]

    return X, y, y_onehot


def parse_timit_line(line):
    start_frame, end_frame, label = line.split(' ')
    logging.debug(start_frame, end_frame, label.strip('\n'))

    return int(start_frame), int(end_frame), label.strip('\n')


def read_labeled_wavfiles(root_timit_path):
    print("Reading from ", root_timit_path)
    wavfiles = sorted(glob.glob(root_timit_path + '/*/*/*.WAV'))
    labels_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))

    logging.debug("Found ", len(wavfiles), " WAV files")
    logging.debug("Found ", len(labels_files), " PHN files")

    X, y = [], []
    for wf, lf in tqdm(zip(wavfiles, labels_files), total=len(wavfiles)):
        mfccs, label = read_labeled_wavfile(wf, lf)
        logging.debug(mfccs)
        logging.debug(label)
        X.append(mfccs)
        y.append(label)

    return np.array(X), np.array(y)

def read_labeled_wavfile(wavfile, labels_file):
    """Map each 20ms recording to a single label."""
    # print("reading ", wavfile, "and: ",labels_file)

    # # output= mfccs_and_deltas, hop_length, n_fft
    # mfccs_and_deltas, segment_duration_frames, hop_duration_frames = utils.wavfile_to_mfccs( wavfile)
    # print mfccs_and_deltas.shape, segment_duration_frames, hop_duration_frames
    # nbFrames = mfccs_and_deltas.shape[1]
    #
    # # Pass through the file with the phones
    # labels = []
    # with open(labels_file, 'r') as f:
    #     for line in f.readlines():
    #         start_frame, end_frame, label = parse_timit_line(line)
    #         #print(start_frame,end_frame)
    #
    #         phn_frames = end_frame - start_frame
    #         labels.extend([label] * phn_frames)
    #
    # print(len(labels), len(labels[0]))
    # print(len(labels), nbFrames)
    #
    # curr_frame = curr_mfcc = 0
    # labeltjes = []
    # mfcctjes = []
    # while (curr_frame < (len(labels) - segment_duration_frames)):
    #     label = max(labels[curr_frame:(curr_frame + segment_duration_frames)])
    #
    #     # print("size mfccs: ",np.size(mfccs_and_deltas[:,curr_mfcc]))
    #     # print("size label: ",np.size(label))
    #     labeltjes.append(label)
    #     mfcctjes.append(mfccs_and_deltas[:, curr_mfcc])
    #
    #     curr_mfcc += 1
    #     curr_frame += hop_duration_frames
    #
    # print(len(labeltjes))
    # print(len(mfcctjes))
    X_val, total_frames = create_mfcc('DUMMY', wavfile)

    from phoneme_set import phoneme_set_39
    phoneme_classes = phoneme_set_39
    y_val = np.zeros(total_frames) - 1
    start_ind = 0
    total_duration = get_total_duration(labels_file)
    fr = open(labels_file)
    for line in fr:
        [start_time, end_time, phoneme] = line.rstrip('\n').split()
        start_time = int(start_time)
        end_time = int(end_time)

        # Convert phoneme strings in y_train to class numbers
        phoneme_num = phoneme_classes[phoneme]
        end_ind = np.round((end_time) / total_duration * total_frames)
        y_val[start_ind:end_ind] = phoneme_num

        start_ind = end_ind
    fr.close()

    # print(X_val.shape, y_val.shape)
    return X_val, y_val


def getMean(X):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(X)
    return scaler.mean_

def apply_normalize(X, mean):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    scaler.mean_ = mean
    X = scaler.fit_transform(X)
    return X

def getFileName(data_dir, name, ext='npy'):
    return os.path.join(data_dir, name + '.%s' % ext)

# MFCC stuff
import scipy.io.wavfile as wav
import python_speech_features
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


def get_total_duration(file):
    """Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
    for line in reversed(list(open(file))):
        [_, val, _] = line.split()
        return int(val)


createPKL(dataset_to_create)