import os
from transform import *
from helpFunctions import copyFilesOfType
from general_tools import *
from phoneme_set import *

nbPhonemes = 39
phoneme_set_list = phoneme_set_39.phoneme_set_39_list  # import list of phonemes,
# convert to dictionary with number mappings (see phoneme_set.py)
values = [i for i in range(0, len(phoneme_set_list))]
phoneme_classes = dict(zip(phoneme_set_list, values))

############### DATA LOCATIONS  ###################
dataPreSplit = True  # some datasets have a pre-defined TEST set (eg TIMIT)
FRAC_VAL = 0.1 # fraction of training data to be used for validation
root = os.path.expanduser("~/TCDTIMIT/audioSR/") # ( keep the trailing slash)
if dataPreSplit:
    dataset = "TIMIT"  # eg TIMIT. You can also manually split up TCDTIMIT according to train/test split in Harte, N.; Gillen, E., "TCD-TIMIT: An Audio-Visual Corpus of Continuous Speech," doi: 10.1109/TMM.2015.2407694
    ## eg TIMIT ##
    dataRootDir       = root+ dataset + "/fixed" + str(nbPhonemes) + os.sep + dataset
    train_source_path = os.path.join(dataRootDir, 'TRAIN')
    test_source_path = os.path.join(dataRootDir, 'TEST')


import numpy as np
from scikits.audiolab import wavread, wavwrite
# merge wav1 and wav2 to out, ratio wav1/wav2 in out is ratio (ratio given in dB)
def mergeAudioFiles(wav1_path, wav2_path, out_path, ratio_dB):
    P2 = 10 ^ (ratio_dB/ 10.0)
    total = 1 + P2
    P2_rel = P2 / total
    P1_rel = 1 / total
    data1, fs1, enc1 = wavread(wav1_path)
    data2, fs2, enc2 = wavread(wav2_path)

    assert fs1 == fs2
    assert enc1 == enc2
    result = P1_rel * data1 + P2_rel * data2

    wavwrite(result, out_path)


def addWhiteNoise(wav_path, out_path, ratio_dB):


import random
def generateBadAudio(outType, srcDir, dstDir, ratio_dB):
    # copy phoneme files
    copyFilesOfType.copyFilesOfType(srcDir, dstDir, ".phn")

    # copy merged wav files
    src_wavs = loadWavs(srcDir)
    for i in tqdm(range(src_wavs)):
        relSrcPath = relpath(srcDir, src_wavs[i]).lstrip("../")
        # print(relSrcPath)
        destPath = os.path.join(dstDir, relSrcPath)
        if outType == 'voices':
            # index of voice to merge
            j = random.randint(0,len(src_wavs)-1)
            mergeAudioFiles(src_wavs[i],src_wavs[j],destPath, ratio_dB)
        else: #add white noise
            addWhiteNoise(src_wavs[i],destPath, ratio_dB)
