import math
import os
import timeit;
program_start_time = timeit.default_timer()
import random
random.seed(int(timeit.default_timer()))

from phoneme_set import phoneme_set_39_list
import general_tools
from preprocessing import *

import logging, formatting
logger = logging.getLogger('PrepTCDTIMIT')
logger.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File logger: see below META VARIABLES


##### SCRIPT META VARIABLES #####
DEBUG = False
debug_size = 50

# TODO:  MODIFY THESE PARAMETERS for other nbPhonemes of mfccTypes. Save location is updated automatically.
nbMFCCs = 39 # 13= just mfcc (13 features). 26 = also derivative (26 features). 39 = also 2nd derivative (39 features)
nbPhonemes = 39
phoneme_set_list = phoneme_set_39_list  # import list of phonemes,
# convert to dictionary with number mappings (see phoneme_set.py)
values = [i for i in range(0, len(phoneme_set_list))]
phoneme_classes = dict(zip(phoneme_set_list, values))

############### DATA LOCATIONS  ###################
dataPreSplit = True #some datasets have a pre-defined TEST set (eg TIMIT)
FRAC_VAL = 0.1 # fraction of training data to be used for validation
root = os.path.expanduser("~/TCDTIMIT/combinedSR/") # ( keep the trailing slash)
if dataPreSplit:
    dataset = "TIMIT" #eg TIMIT. You can also manually split up TCDTIMIT according to train/test split in Harte, N.; Gillen, E., "TCD-TIMIT: An Audio-Visual Corpus of Continuous Speech," doi: 10.1109/TMM.2015.2407694
    ## eg TIMIT ##
    dataRootDir       = root+dataset+"/fixed" + str(nbPhonemes) + os.sep + dataset
    train_source_path = os.path.join(dataRootDir, 'TRAIN')
    test_source_path  = os.path.join(dataRootDir, 'TEST')
    outputDir         = root + dataset + "/binary" + str(nbPhonemes) + os.sep + dataset
else:
    ## just a bunch of wav and phn files, not split up in train and test -> create the split yourself.
    dataset = "TCDTIMIT"
    dataRootDir = root + dataset + "/fixed" + str(nbPhonemes) + "_nonSplit" + os.sep + dataset
    outputDir     = root + dataset + "/binary" + str(nbPhonemes) + os.sep + os.path.basename(dataRootDir)
    FRAC_TRAINING = 0.9  # TOTAL = TRAINING + TEST = TRAIN + VALIDATION + TEST


### store path
target = os.path.join(outputDir, os.path.basename(dataRootDir) + '_' + str(nbMFCCs) + '_ch');
target_path = target + '.pkl'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Already exists, ask if overwrite
if (os.path.exists(target_path)):
    if (not general_tools.queraudio_y_yes_no(target_path + " exists. Overwrite?", "no")):
        raise Exception("Not Overwriting")

# set log file
logFile = outputDir + os.sep + os.path.basename(target) + '.log'
fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

### SETUP ###
if DEBUG:
    logger.info('DEBUG mode: \tACTIVE, only a small dataset will be preprocessed')
    target_path = target + '_DEBUG.pkl'
else:
    logger.info('DEBUG mode: \tDEACTIVE')
    debug_size = None

##### The PREPROCESSING itself #####
logger.info('Preprocessing data ...')

# FIRST, gather the WAV and PHN files, generate MFCCs, extract labels to make inputs and targets for the network
# for a dataset containing no TRAIN/TEST subdivision, just a bunch of wavs -> choose training set yourself
def processDataset(FRAC_TRAINING, data_source_path, logger=None):
    logger.info('  Data: %s ', data_source_path)
    audio_X_all, audio_y_all, valid_frames_all = preprocess_dataset_audio(source_path=data_source_path, nbMFCCs=nbMFCCs, logger=logger, debug=debug_size)
    assert len(audio_X_all) == len(audio_y_all) == len(valid_frames_all)

    logger.info(' Loading data complete.')
    logger.debug('Type and shape/len of audio_X_all')
    logger.debug('type(audio_X_all): {}'.format(type(audio_X_all)))
    logger.debug('type(audio_X_all[0]): {}'.format(type(audio_X_all[0])))
    logger.debug('type(audio_X_all[0][0]): {}'.format(type(audio_X_all[0][0])))
    logger.debug('type(audio_X_all[0][0][0]): {}'.format(type(audio_X_all[0][0][0])))
    logger.info('Creating Validation index ...')

    total_size = len(audio_X_all)  # TOTAL = TRAINING + TEST = TRAIN + VAL + TEST
    total_training_size = int(math.ceil(FRAC_TRAINING * total_size))  # TRAINING = TRAIN + VAL
    test_size = total_size - total_training_size

    # split off a 'test' dataset
    test_idx = random.sample(range(0, total_training_size), test_size)
    test_idx = [int(i) for i in test_idx]
    # ensure that the testidation set isn't empty
    if DEBUG:
        test_idx[0] = 0
        test_idx[1] = 1
    logger.info('Separating test and training set ...')
    audio_X_training = []
    audio_y_training = []
    valid_frames_training = []
    audio_X_test = []
    audio_y_test = []
    valid_frames_test = []
    for i in range(len(audio_X_all)):
        if i in test_idx:
            audio_X_test.append(audio_X_all[i])
            audio_y_test.append(audio_y_all[i])
            valid_frames_test.append(valid_frames_all[i])
        else:
            audio_X_training.append(audio_X_all[i])
            audio_y_training.append(audio_y_all[i])
            valid_frames_training.append(valid_frames_all[i])

    assert len(audio_X_test) == test_size
    assert len(audio_X_training) == total_training_size

    return  audio_X_training, audio_y_training, valid_frames_training, audio_X_test, audio_y_test, valid_frames_test

def processDatasetSplit(train_source_path, test_source_path, logger=None):
    logger.info('  Training data: %s ', train_source_path)
    audio_X_training, audio_y_training, valid_frames_training = preprocess_dataset_audio(source_path=train_source_path, logger=logger,
                                                                                      nbMFCCs=nbMFCCs, debug=debug_size)
    logger.info('  Test data: %s', test_source_path)
    audio_X_test, audio_y_test, valid_frames_test = preprocess_dataset_audio(source_path=test_source_path, logger=logger, nbMFCCs=nbMFCCs, debug=debug_size)
    return audio_X_training, audio_y_training, valid_frames_training, audio_X_test, audio_y_test, valid_frames_test

if dataPreSplit:    audio_X_training, audio_y_training, valid_frames_training, audio_X_test, audio_y_test, valid_frames_test = \
    processDatasetSplit(train_source_path, test_source_path, logger)
else:    audio_X_training, audio_y_training, valid_frames_training, audio_X_test, audio_y_test, valid_frames_test = \
    processDataset(FRAC_TRAINING, dataRootDir, logger)


# SECOND, split off a 'validation' set from the training set. The remainder is the 'train' set
total_training_size = len(audio_X_training)
val_size = int(math.ceil(total_training_size * FRAC_VAL))
train_size = total_training_size - val_size
val_idx = random.sample(range(0, total_training_size), val_size) # choose random indices to be validation data
val_idx = [int(i) for i in val_idx]

logger.info('Length of training')
logger.info("  train X: %s", len(audio_X_training))

# ensure that the validation set isn't empty
if DEBUG:
    val_idx[0] = 0
    val_idx[1] = 1

logger.info('Separating training set into validation and train ...')
audio_X_train = []
audio_y_train = []
valid_frames_train = []
audio_X_val = []
audio_y_val = []
valid_frames_val = []
for i in range(len(audio_X_training)):
    if i in val_idx:
        audio_X_val.append(audio_X_training[i])
        audio_y_val.append(audio_y_training[i])
        valid_frames_val.append(valid_frames_training[i])
    else:
        audio_X_train.append(audio_X_training[i])
        audio_y_train.append(audio_y_training[i])
        valid_frames_train.append(valid_frames_training[i])
assert len(audio_X_val) == val_size

# Print some information
logger.info('Length of train, val, test')
logger.info("  train X: %s", len(audio_X_train))
logger.info("  train y: %s", len(audio_y_train))
logger.info("  train valid_frames: %s", len(valid_frames_train))

logger.info("  val X: %s", len(audio_X_val))
logger.info("  val y: %s", len(audio_y_val))
logger.info("  val valid_frames: %s", len(valid_frames_val))

logger.info("  test X: %s", len(audio_X_test))
logger.info("  test y: %s", len(audio_y_test))
logger.info("  test valid_frames: %s", len(valid_frames_test))


### NORMALIZE data ###
logger.info('Normalizing data ...')
logger.info('    Each channel mean=0, sd=1 ...')

mean_val, std_val, _ = calc_norm_param(audio_X_train)

audio_X_train = normalize(audio_X_train, mean_val, std_val)
audio_X_val = normalize(audio_X_val, mean_val, std_val)
audio_X_test = normalize(audio_X_test, mean_val, std_val)


logger.debug('X train')
logger.debug('  %s %s', type(audio_X_train), len(audio_X_train))
logger.debug('  %s %s', type(audio_X_train[0]), audio_X_train[0].shape)
logger.debug('  %s %s', type(audio_X_train[0][0]), audio_X_train[0][0].shape)
logger.debug('  %s %s', type(audio_X_train[0][0][0]), audio_X_train[0][0].shape)
logger.debug('y train')
logger.debug('  %s %s', type(audio_y_train), len(audio_y_train))
logger.debug('  %s %s', type(audio_y_train[0]), audio_y_train[0].shape)
logger.debug('  %s %s', type(audio_y_train[0][0]), audio_y_train[0][0].shape)


# make sure we're working with float32
audio_X_data_type = 'float32'
audio_X_train = set_type(audio_X_train, audio_X_data_type)
audio_X_val = set_type(audio_X_val, audio_X_data_type)
audio_X_test = set_type(audio_X_test, audio_X_data_type)

audio_y_data_type = 'int32'
audio_y_train = set_type(audio_y_train, audio_y_data_type)
audio_y_val = set_type(audio_y_val, audio_y_data_type)
audio_y_test = set_type(audio_y_test, audio_y_data_type)

valid_frames_data_type = 'int32'
valid_frames_train = set_type(valid_frames_train, valid_frames_data_type)
valid_frames_val = set_type(valid_frames_val, valid_frames_data_type)
valid_frames_test = set_type(valid_frames_test, valid_frames_data_type)


# print some more to check that cast succeeded
logger.debug('X train')
logger.debug('  %s %s', type(audio_X_train), len(audio_X_train))
logger.debug('  %s %s', type(audio_X_train[0]), audio_X_train[0].shape)
logger.debug('  %s %s', type(audio_X_train[0][0]), audio_X_train[0][0].shape)
logger.debug('  %s %s', type(audio_X_train[0][0][0]), audio_X_train[0][0].shape)
logger.debug('y train')
logger.debug('  %s %s', type(audio_y_train), len(audio_y_train))
logger.debug('  %s %s', type(audio_y_train[0]), audio_y_train[0].shape)
logger.debug('  %s %s', type(audio_y_train[0][0]), audio_y_train[0][0].shape)


### STORE DATA ###
logger.info('Saving data to %s', target_path)
dataList = [audio_X_train, audio_y_train, valid_frames_train, audio_X_val, audio_y_val, valid_frames_val, audio_X_test, audio_y_test, valid_frames_test]
general_tools.saveToPkl(target_path, dataList)

# these can be used to evaluate new data, so you don't have to load the whole dataset just to normalize
meanStd_path = os.path.dirname(outputDir) + os.sep + os.path.basename(dataRootDir) + "MeanStd.pkl"
logger.info('Saving Mean and Std_val to %s', meanStd_path)
dataList = [mean_val, std_val]
general_tools.saveToPkl(meanStd_path, dataList)


logger.info('Preprocessing complete!')
logger.info('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))

# helpfunction
from phoneme_set import phoneme_set_39



# LIPREADING files
def depth(path):
    return path.count(os.sep)

def getPhonemeNumberMap():
    return phoneme_set_39

def speakerToBinary(speakerDir, binaryDatabaseDir):
    import numpy as np
    from PIL import Image
    import pickle
    import time

    rootDir = speakerDir
    targetDir = binaryDatabaseDir
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    # get list of images and list of labels (= phonemes)
    images = []
    labels = []
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            name, extension = os.path.splitext(file)
            # copy phoneme files as well
            if extension == ".jpg":
                videoName, frame, phoneme = name.split("_")
                path = ''.join([root, os.sep, file])
                # print(path, " is \t ", phoneme)
                images.append(path)
                labels.append(phoneme)

    # write label and image to binary file, 1 label+image per row
    speakerName = os.path.basename(rootDir)
    outputPath = targetDir + os.sep + speakerName + ".pkl"

    # store in dict with data and 'labelNumber' values.
    rowsize = 120 * 120
    data = np.zeros(shape=(len(images), rowsize), dtype=np.uint8)
    labelNumbers = [0] * len(images)

    print(data.shape)

    for i in range(len(images)):
        label = labels[i]
        image = images[i]

        # for mapping to phonemes (nbClasses = 39)
        phonemeNumberMap = getPhonemeNumberMap()
        labelNumber = phonemeNumberMap[label]  # you could also use the phoneme to viseme map afterwards.

        # for mapping to visemes (nbClasses = 12)
        # phonemeToViseme = getPhonemeToVisemeMap()  # dictionary of phoneme-viseme key-value pairs
        # labelNumber = visemeNumberMap{phonemeToViseme{label}}  # viseme of the phoneme, then get the number of this viseme
        # labelNumbers[i] = labelNumber

        im = np.array(Image.open(image), dtype=np.uint8).flatten()  # flatten to one row per image
        data[i] = im
    # now write python dict to a file
    print("the data file takes: ", data.nbytes, " bytes of memory")
    mydict = {'data': data, 'labels': labelNumbers}
    output = open(outputPath, 'wb')
    pickle.dump(mydict, output, 2)
    output.close()
    print(speakerName, "files have been written to: ", outputPath)
    return 0

def allSpeakersToBinary(databaseDir, binaryDatabaseDir):
    rootDir = databaseDir
    dirList = []
    for dir in directories(rootDir):
        # print(dir)
        # print(relpath(rootDir, dir))
        # print(depth(relpath(rootDir, dir)))
        if depth(relpath(rootDir, dir)) == 1:
            dirList.append(dir)
    print(dirList)
    for speakerDir in dirList:
        print("Extracting files of: ", speakerDir)
        speakerToBinary(speakerDir, binaryDatabaseDir)
    return 0
