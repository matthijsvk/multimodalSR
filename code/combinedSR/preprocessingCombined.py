import timeit;

import numpy as np
from tqdm import tqdm
import os

program_start_time = timeit.default_timer()
import pdb
import logging
from general_tools import *

logger_prepComb = logging.getLogger('combined.prep')
logger_prepComb.setLevel(logging.DEBUG)

# just split the thing up in training and validation set
def getOneSpeaker(speakerFile=None, trainFraction=0.70, validFraction=0.10,
                sourceDataDir=None, storeProcessed=False, processedDir=None,
                verbose=False, loadData=True, viseme=False):

    if processedDir != None:
        store_path = ''.join([processedDir, "_train", str(trainFraction).replace("0.", ""), "valid",
                              str(validFraction).replace("0.", ""), os.sep, os.path.basename(speakerFile)])
        # import pdb;pdb.set_trace()
        # if already processed, just load it from disk
        if os.path.exists(store_path):
            if loadData:  # before starting training, we just want to check if it exists, and generate otherwise. Not load the data
                logger_prepComb.info("loading stored files X's...")
                return unpickle(store_path)
            return
    logger_prepComb.info(" %s processed data doesn't exist yet; generating...", speakerFile)

    # load the images
    # first initialize the matrices

    logger_prepComb.info('loading file %s', speakerFile)
    data = unpickle(sourceDataDir + os.sep + speakerFile)  #    mydict = {'images': allvideosImages, 'mfccs': allvideosMFCCs, 'audioLabels': allvideosAudioLabels, 'validLabels': allvideosValidLabels, 'validAudioFrames': allvideosValidAudioFrames}

    thisN = len(data['images'])
    thisTrain = int(trainFraction * thisN)
    thisValid = int(validFraction * thisN)
    thisTest = thisN - thisTrain - thisValid  # compensates for rounding
    if trainFraction + validFraction == 1.0:
        thisValid = thisN - thisTrain;
        thisTest = 0

    if verbose:
        logger_prepComb.info("This dataset contains %s videos", thisN)
        logger_prepComb.info("train: %s | valid: %s | test: %s", thisTrain, thisValid, thisTest)

    images_train =  list(data['images'][0:thisTrain])
    mfccs_train = list(data['mfccs'][0:thisTrain])
    audioLabels_train = list(data['audioLabels'][0:thisTrain])
    validLabels_train =  list(data['validLabels'][0:thisTrain])
    validAudioFrames_train = list(data['validAudioFrames'][0:thisTrain])

    images_val =  list(data['images'][thisTrain:thisTrain + thisValid])
    mfccs_val = list(data['mfccs'][thisTrain:thisTrain + thisValid])
    audioLabels_val =  list(data['audioLabels'][thisTrain:thisTrain + thisValid])
    validLabels_val = list(data['validLabels'][thisTrain:thisTrain + thisValid])
    validAudioFrames_val =  list(data['validAudioFrames'][thisTrain:thisTrain + thisValid])

    images_test = list(data['images'][thisTrain:thisTrain + thisValid])
    mfccs_test =  list(data['mfccs'][thisTrain:thisTrain + thisValid])
    audioLabels_test = list(data['audioLabels'][thisTrain:thisTrain + thisValid])
    validLabels_test = list(data['validLabels'][thisTrain:thisTrain + thisValid])
    validAudioFrames_test = list(data['validAudioFrames'][thisTrain:thisTrain + thisValid])

    if verbose:
        logger_prepComb.info("nbTrainLoaded: %s", len(images_train))
        logger_prepComb.info("nbValidLoaded: %s", len(images_val))
        logger_prepComb.info("nbTestLoaded: %s", len(images_test))
        logger_prepComb.info("Total loaded: %s", len(images_train) + len(images_val) + len(images_test))

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32' 
    
    if isinstance(images_train, list):              images_train = set_type(images_train, dtypeX);
    if isinstance(mfccs_train, list):               mfccs_train = set_type(mfccs_train,dtypeX);
    if isinstance(audioLabels_train, list):         audioLabels_train = set_type(audioLabels_train,dtypeY);
    if isinstance(validLabels_train, list):         validLabels_train = set_type(validLabels_train,dtypeY);
    if isinstance(validAudioFrames_train, list):    validAudioFrames_train = set_type(validAudioFrames_train,dtypeY);

    if isinstance(images_val, list):                images_val = set_type(images_val,dtypeX);
    if isinstance(mfccs_val, list):                 mfccs_val = set_type(mfccs_val,dtypeX);
    if isinstance(audioLabels_val, list):           audioLabels_val = set_type(audioLabels_val,dtypeY);
    if isinstance(validLabels_val, list):           validLabels_val = set_type(validLabels_val,dtypeY);
    if isinstance(validAudioFrames_val, list):      validAudioFrames_val = set_type(validAudioFrames_val,dtypeY);

    if isinstance(images_test, list):               images_test = set_type(images_test,dtypeX);
    if isinstance(mfccs_test, list):                mfccs_test = set_type(mfccs_test,dtypeX);
    if isinstance(audioLabels_test, list):          audioLabels_test = set_type(audioLabels_test,dtypeY);
    if isinstance(validLabels_test, list):          validLabels_test = set_type(validLabels_test,dtypeY);
    if isinstance(validAudioFrames_test, list):     validAudioFrames_test = set_type(validAudioFrames_test,dtypeY);

    # # add dimension so the audio functions still work
    # for i in range(len(audioLabels_train)):
    #     audioLabels_train[i] = audioLabels_train[i][np.newaxis, ...]
    #     validLabels_train [i]= validLabels_train[i][np.newaxis, ...]
    #     validAudioFrames_train[i] = validAudioFrames_train[i][np.newaxis, ...]
    #
    # # add dimension so the audio functions still work
    # for i in range(len(audioLabels_val)):
    #     audioLabels_val[i] = audioLabels_val[i][np.newaxis, ...]
    #     validLabels_val[i] = validLabels_val[i][np.newaxis, ...]
    #     validAudioFrames_val[i] = validAudioFrames_val[i][np.newaxis, ...]
    #
    # # add dimension so the audio functions still work
    # for i in range(len(audioLabels_test)):
    #     audioLabels_test[i] = audioLabels_test[i][np.newaxis, ...]
    #     validLabels_test[i] = validLabels_test[i][np.newaxis, ...]
    #     validAudioFrames_test[i] = validAudioFrames_test[i][np.newaxis, ...]

    ### STORE DATA ###
    dataList = [[images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train],
                [images_val, mfccs_val, audioLabels_val, validLabels_val, validAudioFrames_val],
                [images_test, mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test]]
    if store_path != None and storeProcessed: saveToPkl(store_path, dataList)

    return dataList
