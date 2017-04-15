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


def preprocess_dataset_audio(source_path, nbMFCCs=39, logger=None, debug=None, verbose=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    X = []
    y = []
    valid_frames = []

    print(nbMFCCs)

    # source_path is the root dir of all the wav/phn files
    wav_files = loadWavs(source_path)
    label_files = loadPhns(source_path)

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
            # calculate frame index with relative length
            # -> works for .phn files with timestamps (TCDTIMIT) and also with audio sample indices (TIMIT)
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


def preprocess_unlabeled_dataset_audio(source_path, verbose=False, logger=None): # TODO
    wav_files = loadWavs(source_path)
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


def load_datasetImages(datapath=os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/database_binary')),
                       trainFraction=0.8,
                       validFraction=0.1, testFraction=0.1, nbClasses=39, onehot=False, type="all", nbLip=1, nbVol=54,
                       verbose=False):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10

    # Lipspeaker 1:                  14627 phonemes,    14617 extacted and useable
    # Lipspeaker 2:  28363 - 14627 = 13736 phonemes     13707 extracted
    # Lipspeaker 3:  42535 - 28363 = 14172 phonemes     14153 extracted
    # total Lipspeakers:  14500 + 13000 + 14000 = 42477

    dtype = 'uint8'
    memAvaliableMB = 6000;
    memAvaliable = memAvaliableMB * 1024
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # prepare data to load
    fnamesLipspkrs = ['Lipspkr%i.pkl' % i for i in range(1, nbLip + 1)]  # all 3 lipsteakers
    fnamesVolunteers = ['Volunteer%i.pkl' % i for i in range(1, nbVol + 1)]  # some volunteers
    if type == "lipspeakers":
        fnames = fnamesLipspkrs
    elif type == "volunteers":
        fnames = fnamesVolunteers
    elif type == "all":
        fnames = fnamesLipspkrs + fnamesVolunteers
    else:
        raise Exception("wrong type of dataset entered")

    datasets = {}
    for name in fnames:
        fname = os.path.join(datapath, name)
        if not os.path.exists(fname):
            raise IOError(fname + " was not found.")
        datasets[name] = cache.datasetCache.cache_file(fname)

    # load the images
    # first initialize the matrices
    train_X = [];
    train_y = []
    valid_X = [];
    valid_y = []
    test_X = [];
    test_y = []

    # now load train data
    trainLoaded = 0
    validLoaded = 0
    testLoaded = 0

    for i, fname in enumerate(fnames):

        if verbose:
            print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded)
            print("nbTrainLoaded: ", trainLoaded)
            print("nbValidLoaded: ", validLoaded)
            print("nbTestLoaded: ", testLoaded)

        print('loading file %s' % datasets[fname])
        data = unpickle(datasets[fname])
        thisN = data['data'].shape[0]
        thisTrain = int(trainFraction * thisN)
        thisValid = int(validFraction * thisN)
        thisTest = thisN - thisTrain - thisValid  # compensates for rounding\
        if verbose:
            print("This dataset contains ", thisN, " images")
            print("now loading : nbTrain, nbValid, nbTest")
            print("              ", thisTrain, thisValid, thisTest)

        train_X = train_X + list(data['data'][0:thisTrain])
        valid_X = valid_X + list(data['data'][thisTrain:thisTrain + thisValid])
        test_X = test_X + list(data['data'][thisTrain + thisValid:thisN])

        train_y = train_y + list(data['labels'][0:thisTrain])
        valid_y = valid_y + list(data['labels'][thisTrain:thisTrain + thisValid])
        test_y = test_y + list(data['labels'][thisTrain + thisValid:thisN])

        trainLoaded += thisTrain
        validLoaded += thisValid
        testLoaded += thisTest
        if verbose:
            print("nbTrainLoaded: ", trainLoaded)
            print("nbValidLoaded: ", validLoaded)
            print("nbTestLoaded: ", testLoaded)
            print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded)

        # estimate as float32 = 4* memory as uint8
        memEstimate = 4 * (sys.getsizeof(train_X) + sys.getsizeof(valid_X) + sys.getsizeof(test_X) + \
                           sys.getsizeof(train_y) + sys.getsizeof(valid_y) + sys.getsizeof(test_y))
        if verbose: print("memory estaimate: ", memEstimate / 1000.0, "MB")
        if memEstimate > 0.6 * memAvaliable:
            print("loaded too many for memory, stopping loading...")
            break

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32'  # needed for
    if isinstance(train_X, list):       train_X = np.asarray(train_X).astype(dtypeX);
    if isinstance(train_y, list):       train_y = np.asarray(train_y).astype(dtypeY);
    if isinstance(valid_X, list):       valid_X = np.asarray(valid_X).astype(dtypeX);
    if isinstance(valid_y, list):       valid_y = np.asarray(valid_y).astype(dtypeY);
    if isinstance(test_X, list):        test_X = np.asarray(test_X).astype(dtypeX);
    if isinstance(test_y, list):        test_y = np.asarray(test_y).astype(dtypeY);

    if verbose:
        print("TRAIN: ", train_X.shape, train_X[0][0].dtype)
        print(train_y.shape, train_y[0].dtype)
        print("VALID: ", valid_X.shape)
        print(valid_y.shape)
        print("TEST: ", test_X.shape)
        print(test_y.shape)

    memTot = train_X.nbytes + valid_X.nbytes + test_X.nbytes + train_y.nbytes + valid_y.nbytes + test_y.nbytes
    print("Total memory size required as float32: ", memTot / 1000000, " MB")

    # fix labels (labels start at 1, but the library expects them to start at 0)
    train_y = train_y - 1
    valid_y = valid_y - 1
    test_y = test_y - 1

    # rescale to interval [-1,1], cast to float32 for GPU use
    train_X = np.multiply(2. / 255., train_X, dtype='float32')
    train_X = np.subtract(train_X, 1., dtype='float32');
    valid_X = np.multiply(2. / 255., valid_X, dtype='float32')
    valid_X = np.subtract(valid_X, 1., dtype='float32');
    test_X = np.multiply(2. / 255., test_X, dtype='float32')
    test_X = np.subtract(test_X, 1., dtype='float32');

    if verbose:
        print("Train: ", train_X.shape, train_X[0][0].dtype)
        print("Valid: ", valid_X.shape, valid_X[0][0].dtype)
        print("Test: ", test_X.shape, test_X[0][0].dtype)

    # reshape to get one image per row
    train_X = np.reshape(train_X, (-1, 1, 120, 120))
    valid_X = np.reshape(valid_X, (-1, 1, 120, 120))
    test_X = np.reshape(test_X, (-1, 1, 120, 120))

    # also flatten targets to get one target per row
    # train_y = np.hstack(train_y)
    # valid_y = np.hstack(valid_y)
    # test_y = np.hstack(test_y)

    # Onehot the targets
    if onehot:
        train_y = np.float32(np.eye(nbClasses)[train_y])
        valid_y = np.float32(np.eye(nbClasses)[valid_y])
        test_y = np.float32(np.eye(nbClasses)[test_y])

    # for hinge loss
    train_y = 2 * train_y - 1.
    valid_y = 2 * valid_y - 1.
    test_y = 2 * test_y - 1.

    # cast to correct datatype, just to be sure. Everything needs to be float32 for GPU processing
    dtypeX = 'float32'
    dtypeY = 'int32'
    train_X = train_X.astype(dtypeX);
    train_y = train_y.astype(dtypeY);
    valid_X = valid_X.astype(dtypeX);
    valid_y = valid_y.astype(dtypeY);
    test_X = test_X.astype(dtypeX);
    test_y = test_y.astype(dtypeY);
    if verbose:
        print("\n Final datatype: ")
        print("TRAIN: ", train_X.shape, train_X[0][0].dtype)
        print(train_y.shape, train_y[0].dtype)
        print("VALID: ", valid_X.shape)
        print(valid_y.shape)
        print("TEST: ", test_X.shape)
        print(test_y.shape)

    return train_X, train_y, valid_X, valid_y, test_X, test_y
