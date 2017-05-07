import os
import logging, formatting
logger_combined = logging.getLogger('combined')
logger_combined.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))


# This preprocess the lipspeakers only, and stores all dataa in 3 files 1 train, 1 validation, 1 test
# These files can be loaded into memory once, and stay there so you don't need to wait for loading/unloading data from memory
# like with the full dataset or the volunteers. This can help you get results faster.
# To use this, set dataType to "lipspeakers" in combinedNN.py.

dataset = "TCDTIMIT"
root_dir = os.path.expanduser("~/TCDTIMIT/combinedSR/")
store_dir = root_dir + dataset + "/results"
if not os.path.exists(store_dir): os.makedirs(store_dir)

if not os.path.exists(store_dir): os.makedirs(store_dir)
database_binaryDir = root_dir + dataset + '/binary'
processedDir = database_binaryDir + "_finalProcessed"
datasetType = "combined";


lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];

## Get images per video for lipspeakers
import preprocessingCombined

allImages_train = []; allMfccs_train = []; allAudioLabels_train=[]; allValidLabels_train=[]; allValidAudioFrames_train = []
allImages_val = []; allMfccs_val = []; allAudioLabels_val=[]; allValidLabels_val=[]; allValidAudioFrames_val = []
allImages_test = []; allMfccs_test = []; allAudioLabels_test=[]; allValidLabels_test=[]; allValidAudioFrames_test = []

for lipspeaker in lipspeakers:
    train, val, test = preprocessingCombined.getOneSpeaker(lipspeaker,
                                                           sourceDataDir=database_binaryDir,
                                                           storeProcessed=False, processedDir=processedDir,
                                                           trainFraction=0.7, validFraction=0.1, verbose=False)
    images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train = train
    images_val, mfccs_val, audioLabels_val, validLabels_val, validAudioFrames_val = val
    images_test, mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test = test

    allImages_train += images_train
    allMfccs_train += mfccs_train
    allAudioLabels_train += audioLabels_train
    allValidLabels_train += validLabels_train
    allValidAudioFrames_train += validAudioFrames_train

    allImages_val += images_val
    allMfccs_val += mfccs_val
    allAudioLabels_val += audioLabels_val
    allValidLabels_val += validLabels_val
    allValidAudioFrames_val += validAudioFrames_val

    allImages_test += images_test
    allMfccs_test += mfccs_test
    allAudioLabels_test += audioLabels_test
    allValidLabels_test += validLabels_test
    allValidAudioFrames_test += validAudioFrames_test

#import pdb;pdb.set_trace()
from general_tools import *

storePath = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTrain.pkl")

saveToPkl(storePath, [allImages_train,
                      allMfccs_train ,
                      allAudioLabels_train,
                      allValidLabels_train,
                      allValidAudioFrames_train ])

storePath = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersVal.pkl")

saveToPkl(storePath, [allImages_val,
                      allMfccs_val,
                      allAudioLabels_val,
                      allValidLabels_val,
                      allValidAudioFrames_val])

storePath = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTest.pkl")

saveToPkl(storePath, [allImages_test,
                      allMfccs_test,
                      allAudioLabels_test,
                      allValidLabels_test,
                      allValidAudioFrames_test])
