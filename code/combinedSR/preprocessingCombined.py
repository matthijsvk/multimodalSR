import timeit;

import numpy as np
from tqdm import tqdm
import os

program_start_time = timeit.default_timer()
import pdb
import logging
from general_tools import *

logger = logging.getLogger('combined.prep')
logger.setLevel(logging.DEBUG)

# just split the thing up in training and validation set
def getOneSpeaker(speakerFile=None, trainFraction=0.70, validFraction=0.10,
                  sourceDataDir=None, storeProcessed=False, processedDir=None,
                  verbose=False, loadData=True, viseme=False, nbPhonemes=39,
                  withNoise = False, noiseType = 'white', ratio_dB = -3, dataset="TCDTIMIT",
                  logger=logger):

    store_path = None
    if processedDir != None:
        store_path = ''.join([processedDir, "_train", str(trainFraction).replace("0.", ""), "valid",
                              str(validFraction).replace("0.", ""), os.sep, os.path.basename(speakerFile).replace(".pkl","")+("_viseme" if viseme else "")+".pkl"])
        # if already processed, just load it from disk
        if os.path.exists(store_path):
            if loadData:  # before starting training, we just want to check if it exists, and generate otherwise. Not load the data
                if verbose: logger.info("loading stored files X's...")
                train, val, test = unpickle(store_path)
                if withNoise: #only works for loadPerSpeaker, for lipspeakers you have to generate test audio with datasetToPkl_lipspeakers.py
                    audio_data_path = os.path.expanduser("~/TCDTIMIT/combinedSR/") + dataset + "/binaryAudio" + str(
                        nbPhonemes) + "_" + noiseType + os.sep + "ratio" + str(ratio_dB) + os.sep + speakerFile
                    test[1:] = unpickle(audio_data_path)
                return [train,val,test]
    logger.info(" %s processed data doesn't exist yet; generating...", speakerFile)


    logger.info('loading file %s', speakerFile)
    data = unpickle(sourceDataDir + os.sep + speakerFile)  #    mydict = {'images': allvideosImages, 'mfccs': allvideosMFCCs, 'audioLabels': allvideosAudioLabels, 'validLabels': allvideosValidLabels, 'validAudioFrames': allvideosValidAudioFrames}

    thisN = len(data['images'])
    thisTrain = int(trainFraction * thisN)
    thisValid = int(validFraction * thisN)
    thisTest = thisN - thisTrain - thisValid  # compensates for rounding
    if trainFraction + validFraction == 1.0:
        thisValid = thisN - thisTrain;
        thisTest = 0

    if verbose:
        logger.info("This dataset contains %s videos", thisN)
        logger.info("train: %s | valid: %s | test: %s", thisTrain, thisValid, thisTest)

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

    images_test = list(data['images'][thisTrain + thisValid:thisN])
    mfccs_test =  list(data['mfccs'][thisTrain + thisValid:thisN])
    audioLabels_test = list(data['audioLabels'][thisTrain + thisValid:thisN])
    validLabels_test = list(data['validLabels'][thisTrain + thisValid:thisN])
    validAudioFrames_test = list(data['validAudioFrames'][thisTrain + thisValid:thisN])

    if viseme: #convert all labels from phoneme to viseme
        from phoneme_set import phonemeToViseme, viseme_set, classToPhoneme39  # dictionary of phoneme-viseme key-value pairs
        for labelList in [validLabels_train, validLabels_val, validLabels_test]:
            for videoNr in range(len(labelList)):
                for frameNr in range(len(labelList[videoNr])):
                    label = labelList[videoNr][frameNr]
                    phoneme = classToPhoneme39[label]
                    viseme = phonemeToViseme[phoneme]
                    labelList[videoNr][frameNr] = viseme_set[viseme]

    if verbose:
        logger.info("nbTrainLoaded: %s", len(images_train))
        logger.info("nbValidLoaded: %s", len(images_val))
        logger.info("nbTestLoaded: %s", len(images_test))
        logger.info("Total loaded: %s", len(images_train) + len(images_val) + len(images_test))

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

    ### STORE DATA ###
    dataList = [[images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train],
                [images_val, mfccs_val, audioLabels_val, validLabels_val, validAudioFrames_val],
                [images_test, mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test]]

    import pdb;pdb.set_trace()
    if store_path != None and storeProcessed: saveToPkl(store_path, dataList)

    return dataList
