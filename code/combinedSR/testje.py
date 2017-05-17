import os,sys
from general_tools import *

datasetName = "TCDTIMIT"

withNoise = True
noiseType = 'white'
ratio_dB = -10

original = unpickle(
        os.path.expanduser("~/TCDTIMIT/combinedSR/TCDTIMIT/binaryLipspeakers/allLipspeakersTest.pkl"))
allImages_test, allMfccs_test, allAudioLabels_test, allValidLabels_test, allValidAudioFrames_test = original
for el in original: assert len(el) == len(original[0]);
print(len(original[0]))

print("images:", allImages_test[0].shape)
print("validLabels:", allValidLabels_test[0].shape)
import pdb;pdb.set_trace()

if withNoise:
    testDataPath = os.path.expanduser("~/TCDTIMIT/combinedSR/") + datasetName + "/binaryLipspeakers" + os.sep \
                   + 'allLipspeakersTest' + "_" + noiseType + "_" + "ratio" + str(ratio_dB) + '.pkl'
    new = unpickle(testDataPath)
    for el in new: assert len(el) == len(new[0]);
    print(len(new[0]))

    allMfccs_test, allAudioLabels_test, allValidLabels_test, allValidAudioFrames_test = new

import pdb;pdb.set_trace()
