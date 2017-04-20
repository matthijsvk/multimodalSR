import timeit;

import numpy as np
from tqdm import tqdm
import os

program_start_time = timeit.default_timer()
import pdb
import logging
from general_tools import *

logger_prepComb = logging.getLogger('lipreading.prep')
logger_prepComb.setLevel(logging.DEBUG)


def getOneSpeaker(speakerFile, trainFraction, validFraction, storeDir = None, verbose=False):
    store_path = storeDir + os.sep + speakerFile.replace(".pkl","") + "_split_train"+str(trainFraction)+"_valid"+str(validFraction)+"_test"+str(testFraction)+".pkl" 
    
    # load the images
    # first initialize the matrices
    images_train = [];
    mfccs_train = []
    audioLabels_train = []
    validLabels_train = []
    validAudioFrames_train = []

    images_val = [];
    mfccs_val = []
    audioLabels_val = []
    validLabels_val = []
    validAudioFrames_val = []

    images_test = [];
    mfccs_test = []
    audioLabels_test = []
    validLabels_test = []
    validAudioFrames_test = []
    
    # TODO: add masks here as well?  depends on batch size...

    logger_prepComb.info('loading file %s', speakerFile)
    data = unpickle(speakerFile)  #    mydict = {'images': allvideosImages, 'mfccs': allvideosMFCCs, 'audioLabels': allvideosAudioLabels, 'validLabels': allvideosValidLabels, 'validAudioFrames': allvideosValidAudioFrames}

    thisN = data['images'].shape[0]
    thisTrain = int(trainFraction * thisN)
    thisValid = int(validFraction * thisN)
    thisTest = thisN - thisTrain - thisValid  # compensates for rounding
    if trainFraction + validFraction == 1.0:
        thisValid = thisN - thisTrain;
        thisTest = 0

    if verbose:
        logger_prepComb.info("This dataset contains %s images_train", thisN)
        logger_prepComb.info("now loading : nbTrain, nbValid, nbTest")
        logger_prepComb.info("\t\t\t %s %s %s", thisTrain, thisValid, thisTest)

    images_train = images_train + list(data['images'][0:thisTrain])
    mfccs_train = mfccs_train + list(data['mfccs'][0:thisTrain])
    audiolabels_train = audioLabels_train + list(data['audioLabels'][0:thisTrain])
    validLabels_train = validLabels_train + list(data['validLabels'][0:thisTrain])
    validAudioFrames_train = validAudioFrames_train + list(data['validAudioFrames'][0:thisTrain])

    images_val = images_val + list(data['images'][thisTrain:thisTrain + thisValid])
    mfccs_val = mfccs_val + list(data['mfccs'][thisTrain:thisTrain + thisValid])
    audiolabels_val = audioLabels_val + list(data['audioLabels'][thisTrain:thisTrain + thisValid])
    validLabels_val = validLabels_val + list(data['validLabels'][thisTrain:thisTrain + thisValid])
    validAudioFrames_val = validAudioFrames_val + list(data['validAudioFrames'][thisTrain:thisTrain + thisValid])

    images_test = images_test + list(data['images'][thisTrain:thisTrain + thisValid])
    mfccs_test = mfccs_test + list(data['mfccs'][thisTrain:thisTrain + thisValid])
    audiolabels_test = audioLabels_test + list(data['audioLabels'][thisTrain:thisTrain + thisValid])
    validLabels_test = validLabels_test + list(data['validLabels'][thisTrain:thisTrain + thisValid])
    validAudioFrames_test = validAudioFrames_test + list(data['validAudioFrames'][thisTrain:thisTrain + thisValid])


    if verbose:
        logger_prepComb.info("nbTrainLoaded: ", len(images_train))
        logger_prepComb.info("nbValidLoaded: ", len(images_val))
        logger_prepComb.info("nbTestLoaded: ", len(images_test))
        logger_prepComb.info("Total loaded: ", len(images_train) + len(images_val) + len(images_test))

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32' 
    
    if isinstance(images_train, list):       images_train = np.asarray(images_train).astype(dtypeX);
    if isinstance(mfccs_train, list):       mfccs_train = np.asarray(mfccs_train).astype(dtypeX);
    if isinstance(audioLabels_train, list):       audioLabels_train = np.asarray(audioLabels_train).astype(dtypeY);
    if isinstance(validLabels_train, list):       validLabels_train = np.asarray(validLabels_train).astype(dtypeY);
    if isinstance(validAudioFrames_train, list):       validAudioFrames_train = np.asarray(validAudioFrames_train).astype(dtypeY);

    if isinstance(images_val, list):       images_val = np.asarray(images_val).astype(dtypeX);
    if isinstance(mfccs_val, list):       mfccs_val = np.asarray(mfccs_val).astype(dtypeX);
    if isinstance(audioLabels_val, list):       audioLabels_val = np.asarray(audioLabels_val).astype(dtypeY);
    if isinstance(validLabels_val, list):       validLabels_val = np.asarray(validLabels_val).astype(dtypeY);
    if isinstance(validAudioFrames_val, list):       validAudioFrames_val = np.asarray(validAudioFrames_val).astype(dtypeY);

    if isinstance(images_test, list):       images_test = np.asarray(images_test).astype(dtypeX);
    if isinstance(mfccs_test, list):       mfccs_test = np.asarray(mfccs_test).astype(dtypeX);
    if isinstance(audioLabels_test, list):       audioLabels_test = np.asarray(audioLabels_test).astype(dtypeY);
    if isinstance(validLabels_test, list):       validLabels_test = np.asarray(validLabels_test).astype(dtypeY);
    if isinstance(validAudioFrames_test, list):       validAudioFrames_test = np.asarray(validAudioFrames_test).astype(dtypeY);

    if isinstance(validLabels_train, list):       validLabels_train = np.asarray(validLabels_train).astype(dtypeY);

    ### STORE DATA ###
    dataList = [[images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train],
                [images_val, mfccs_val, audioLabels_val, validLabels_val, validAudioFrames_val],
                [images_test, mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test]]
    if store_path != None: saveToPkl(store_path, dataList)

    return dataList
