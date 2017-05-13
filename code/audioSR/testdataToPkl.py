import timeit;

program_start_time = timeit.default_timer()
import random

random.seed(int(timeit.default_timer()))

from general_tools import *

import logging
from audioSR import formatting, preprocessWavs

logger = logging.getLogger('PrepTCDTIMIT')
logger.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


nbMFCCs = 39  # 13= just mfcc (13 features). 26 = also derivative (26 features). 39 = also 2nd derivative (39 features)
nbPhonemes = 39

############### DATA LOCATIONS  ###################
root = os.path.expanduser("~/TCDTIMIT/audioSR/")  # ( keep the trailing slash)
dataset = "TIMIT"
forceOverwrite = False
###################
normalizePkl_path = root + dataset + "/binary" + str(nbPhonemes) + os.sep + dataset + "MeanStd.pkl"

def main():
    noiseTypes = ['white','voices']
    ratio_dBs = [0, -3, -5, -10]

    for noiseType in noiseTypes:
        for ratio_dB in ratio_dBs:
            oneTypeToPkl(noiseType, ratio_dB)

def oneTypeToPkl(noiseType, ratio_dB):
    global target_path, dataList
    dataRootDir = root + dataset + "/fixed" + str(nbPhonemes) + "_" + noiseType + os.sep + "ratio" + str(
        ratio_dB) + os.sep + 'TEST'
    outputDir = root + dataset + "/binary" + str(nbPhonemes) + "_" + noiseType + \
                os.sep + "ratio" + str(ratio_dB) + os.sep + dataset
    FRAC_TRAINING = 0.0  # TOTAL = TRAINING + TEST = TRAIN + VALIDATION + TEST
    ### store path
    target = os.path.join(outputDir, dataset + '_' + str(nbMFCCs) + '_ch');
    target_path = target + '.pkl'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Already exists, ask if overwrite
    if (os.path.exists(target_path)):
        if not forceOverwrite:
            logger.info("This file already exists, skipping", target_path)
            return 0

    ##### The PREPROCESSING itself #####
    logger.info('Preprocessing data ...')

    # FIRST, gather the WAV and PHN files, generate MFCCs, extract labels to make inputs and targets for the network
    # for a dataset containing no TRAIN/TEST subdivision, just a bunch of wavs -> choose training set yourself
    def processDataset(FRAC_TRAINING, data_source_path, logger=None):
        logger.info('  Data: %s ', data_source_path)
        X_test, y_test, valid_frames_test = preprocessWavs.preprocess_dataset(source_path=data_source_path,
                                                                              nbMFCCs=nbMFCCs,
                                                                              logger=logger, debug=None)
        assert len(X_test) == len(y_test) == len(valid_frames_test)

        logger.info(' Loading data complete.')
        logger.debug('Type and shape/len of X_test')
        logger.debug('type(X_test): {}'.format(type(X_test)))
        logger.debug('type(X_test[0]): {}'.format(type(X_test[0])))
        logger.debug('type(X_test[0][0]): {}'.format(type(X_test[0][0])))
        logger.debug('type(X_test[0][0][0]): {}'.format(type(X_test[0][0][0])))

        return X_test, y_test, valid_frames_test

    X_test, y_test, valid_frames_test = processDataset(FRAC_TRAINING, dataRootDir, logger)
    logger.info("  test X: %s", len(X_test))
    logger.info("  test y: %s", len(y_test))
    logger.info("  test valid_frames: %s", len(valid_frames_test))
    ### NORMALIZE data ###
    logger.info('Normalizing data ...')
    logger.info('    Each channel mean=0, sd=1 ...')
    mean_val, std_val = unpickle(normalizePkl_path)
    X_test = preprocessWavs.normalize(X_test, mean_val, std_val)
    # make sure we're working with float32
    X_data_type = 'float32'
    X_test = preprocessWavs.set_type(X_test, X_data_type)
    y_data_type = 'int32'
    y_test = preprocessWavs.set_type(y_test, y_data_type)
    valid_frames_data_type = 'int32'
    valid_frames_test = preprocessWavs.set_type(valid_frames_test, valid_frames_data_type)
    # print some more to check that cast succeeded
    logger.debug('X test')
    logger.debug('  %s %s', type(X_test), len(X_test))
    logger.debug('  %s %s', type(X_test[0]), X_test[0].shape)
    logger.debug('  %s %s', type(X_test[0][0]), X_test[0][0].shape)
    logger.debug('  %s %s', type(X_test[0][0][0]), X_test[0][0].shape)
    logger.debug('y test')
    logger.debug('  %s %s', type(y_test), len(y_test))
    logger.debug('  %s %s', type(y_test[0]), y_test[0].shape)
    logger.debug('  %s %s', type(y_test[0][0]), y_test[0][0].shape)
    ### STORE DATA ###
    logger.info('Saving data to %s', target_path)
    dataList = [X_test, y_test, valid_frames_test]
    saveToPkl(target_path, dataList)

    logger.info('Preprocessing complete!')
    logger.info('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))

if __name__ == "__main__":
    main()
