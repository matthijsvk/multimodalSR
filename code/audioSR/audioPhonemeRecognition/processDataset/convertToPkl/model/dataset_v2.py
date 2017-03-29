import glob
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
import utils
from sklearn import preprocessing

"""
This file is based on https://github.com/dtjchen/spoken-command-processor
"""

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
logging.debug('A debug message!')

train_dataset_path = os.environ['TIMIT_TRAINING_PATH']
test_dataset_path = os.environ['TIMIT_TESTING_PATH']
data_root = os.environ['MODEL_PARAMETERS']

print("initialize Speech2Phonemes reader...")

data_dir = os.path.join(data_root, 'speech2phonemes26Mels')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
print("data dir: ", data_dir)

            
def getFileName(data_dir, name, ext='npy'):
    return os.path.join(data_dir, name + '.%s' % ext)


def getMean(X):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(X)
    return scaler.mean_


def apply_normalize(X, mean):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    scaler.mean_ = mean
    X = scaler.fit_transform(X)
    return X

def createPKL(type):
    import pickle
    print("Generating ", type, " files from dataset...")

    if type == 'all':
        createPKL('train')
        createPKL('test')
        return 0
    elif type == 'train':
        outputPath = getFileName(data_dir, 'trainData')
        print(outputPath)
        if (os.path.exists(outputPath)):
            if (not utils.query_yes_no("TrainData.npy exists. Overwrite?","no")):
                return 0
        X, y = read_labeled_wavfiles(train_dataset_path)
        mean_val = getMean(X)
        np.save(getFileName(data_dir, 'Train_mean'), np.array([mean_val]))
        X = apply_normalize(X, mean_val)
    elif type == 'test':
        outputPath = getFileName(data_dir, 'testData')
        if (os.path.exists(outputPath)):
            if (not utils.query_yes_no("TestData.npy exists. Overwrite?", "no")):
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
        i = 0
        for mfccs, label in read_labeled_wavfile(wf, lf):
            logging.debug(mfccs)
            logging.debug(label)
            X.append(mfccs)
            y.append(label)
            i += 1
            # if i==3: import pdb; pdb.set_trace()

    # Convert phoneme strings in y_train to class numbers
    from phoneme_set import phoneme_set_39
    phoneme_classes = phoneme_set_39
    y = [phoneme_classes[y[i]] for i in range(len(y))]

    return np.array(X), np.array(y)

def read_labeled_wavfile(wavfile, labels_file):
    """Map each 20ms recording to a single label."""
    # print("reading ", wavfile, "and: ",labels_file)

    # output= mfccs_and_deltas, hop_length, n_fft
    mfccs_and_deltas, segment_duration_frames, hop_duration_frames = utils.wavfile_to_mfccs( wavfile)

    # Pass through the file with the phones
    labels = []
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            start_frame, end_frame, label = parse_timit_line(line)
            # logging.debug(start_frame,end_frame)

            phn_frames = end_frame - start_frame
            labels.extend([label] * phn_frames)

    curr_frame = curr_mfcc = 0

    while (curr_frame < (len(labels) - segment_duration_frames)):
        label = max(labels[curr_frame:(curr_frame + segment_duration_frames)])

        # print("size mfccs: ",np.size(mfccs_and_deltas[:,curr_mfcc]))
        # print("size label: ",np.size(label))
        yield mfccs_and_deltas[:, curr_mfcc], label

        curr_mfcc += 1
        curr_frame += hop_duration_frames


def getMean(X):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(X)
    return scaler.mean_

def apply_normalize(X, mean):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    scaler.mean_ = mean
    X = scaler.fit_transform(X)
    return X