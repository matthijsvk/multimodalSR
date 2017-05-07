import numpy as np
import sys, os
import cPickle
import general_tools

import logging
import formatting

logger_prepLip = logging.getLogger('prepLip')
logger_prepLip.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_prepLip.addHandler(ch)


from general_tools import *

lipspkr_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTrain.pkl")
train = unpickle(lipspkr_path)


lipspkr_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersVal.pkl")
val = unpickle(lipspkr_path)

lipspkr_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTest.pkl")
test = unpickle(lipspkr_path)

images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train = train
images_val, mfccs_val, audioLabels_val, validLabels_val, validAudioFrames_val = val
images_test, mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test = test


# drop everything we don't need for lipreading
trainLip = [images_train, validLabels_train]
valLip = [images_val, validLabels_val]
testLip = [images_test, validLabels_test]


logger_prepLip.info("TRAIN: %s %s", len(images_train), images_train[0][0].dtype)
logger_prepLip.info("%s %s", len(validLabels_train), validLabels_train[0].dtype)
logger_prepLip.info("VALID: %s", len(images_val))
logger_prepLip.info("%s", len(validLabels_val))
logger_prepLip.info("TEST: %s", len(images_test))
logger_prepLip.info("%s", len(validLabels_test))


def flattenVideos(images, labels):
    assert len(images) == len(labels)
    images_new = images[0]
    labels_new = labels[0]
    for i in range(1,len(images)):
        images_new = np.concatenate((images_new, images[i]))
        labels_new = np.concatenate((labels_new, labels[i]))
    return images_new, labels_new

X_train, y_train = flattenVideos(images_train, validLabels_train)
train_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTrain.pkl")
saveToPkl(train_path, [X_train, y_train])

X_val, y_val = flattenVideos(images_val, validLabels_val)
val_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersVal.pkl")
saveToPkl(val_path, [X_val, y_val])

X_test, y_test = flattenVideos(images_test, validLabels_test)
test_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTest.pkl")
saveToPkl(test_path, [X_test, y_test])

        