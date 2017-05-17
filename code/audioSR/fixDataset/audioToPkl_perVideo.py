import timeit;

from tqdm import tqdm
import sys, os
sys.path.insert(0, os.path.abspath('../'))
print(os.path.abspath('../'))

program_start_time = timeit.default_timer()
import random
random.seed(int(timeit.default_timer()))

# label files are in audio frames @16kHz, not in seconds
from phoneme_set import phoneme_set_39

import math
import cPickle

import logging
import formatting

logger_combinedPrep = logging.getLogger('prepcombined')
logger_combinedPrep.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger_combinedPrep.addHandler(ch)

from general_tools import *
from preprocessWavs import *

nbMFCCs = 39
nbPhonemes = 39

def main():
    # This file can be used to generate the audio arrays for use in combinedSR (replace the clean audio with noisy audio)
    # If you don't want to use noisy audio, just use combinedSR/datasetToPkl.py
    # Only works for test set for now, but should be easily adaptable for train set as well if you want to train on noisy data
    # In that case you'll need to generate noisy audio using mergeAudioFiles.py in audioSR/fixDataset/

    # when training/evaluating on combinedSR, when

    dataset = "TCDTIMIT"
    noiseTypes = ['voices', 'white']
    ratio_dBs = [0, -3, -5, -10]

    for noiseType in noiseTypes:
        for ratio_dB in ratio_dBs:
            databaseDir = os.path.expanduser("~/TCDTIMIT/audioSR/") + dataset + "/fixed" + str(nbPhonemes) \
                          + "_" + noiseType + os.sep + "ratio" + str(ratio_dB) + "/lipspeakers"

            outputDir = os.path.expanduser("~/TCDTIMIT/combinedSR/") + dataset + "/binaryAudio" + str(nbPhonemes) \
                        + "_" + noiseType + os.sep + "ratio" + str(ratio_dB)

            meanStdAudio = unpickle("../database_averages/TCDTIMITMeanStd.pkl")
            allSpeakersToBinary(databaseDir, outputDir, meanStdAudio, test=False, overWrite=False)


def audioDirToArrays(dir, nbMFCCs=39, verbose=False):
    thisValidAudioFrames = []  # these are the audio valid frames of this video
    labels_fromAudio = []
    # only 1 mfcc file, no need for a list

    for root, dirs, files in os.walk(dir):  # files in videoDir, should only be .jpg, .wav and .phn (and .vphn)
        for file in files:
            name, extension = os.path.splitext(file)

            if extension == ".wav":
                # Get MFCC of the WAV
                wav_path = ''.join([root, os.sep, file])
                X_val, total_frames = create_mfcc('DUMMY', wav_path, nbMFCCs)
                total_frames = int(total_frames)
                thisMFCCs = X_val

                # find the corresponding phoneme file, get the phoneme classes
                phn_path = ''.join([root, os.sep, name, ".phn"])
                if not os.path.exists(phn_path):
                    raise IOError("Phoneme files ", phn_path, " not found...")
                    import pdb;                    pdb.set_trace()

                # some .PHN files don't start at 0. Set default phoneme to silence (expected at the end of phoneme_set_list)
                labels_fromAudio = total_frames * [phoneme_set_39_list[-1]]

                fr = open(phn_path)
                total_duration = get_total_duration(phn_path)

                for line in fr:
                    [start_time, end_time, phoneme] = line.rstrip('\n').split()

                    # check that phoneme is found in dict
                    if (phoneme not in phoneme_set_39_list):
                        logger.error("In file: %s, phoneme not found: %s", phn_path, phoneme)
                        pdb.set_trace()
                    # fill all the labels with the correct phoneme for each timestep
                    start_ind = int(np.round(int(start_time) * total_frames / float(total_duration)))
                    end_ind = int(np.round(int(end_time) * total_frames / float(total_duration)))
                    labels_fromAudio[start_ind:end_ind] = (end_ind - start_ind) * [phoneme]

                    # get valid audio frame
                    valid_ind = int(np.round( (start_ind + end_ind) * 0.5))
                    thisValidAudioFrames.append(valid_ind)

                    # get video frames like in TCDTIMITprocessing,  sanity check
                    start = float(start_time) / 16000  # 16kHz is audio sampling frequency; label files are in audio frames, not in seconds
                    end = float(end_time) / 16000
                    extractionTime = start * (1 - 0.5) + (0.5) * end
                    frame = int(math.floor(extractionTime * 29.97))  # convert to

                    if verbose:
                        logger.debug('%s', (total_frames / float(total_duration)))
                        logger.debug('TIME  start: %s end: %s, phoneme: %s, class: %s', start_time, end_time,
                                     phoneme, phoneme_num)
                        logger.debug('FRAME start: %s end: %s, phoneme: %s, class: %s', start_ind, end_ind,
                                     phoneme, phoneme_num)
                fr.close()

                # save only the valid labels to this list
                try:
                    validLabels_fromAudio = [labels_fromAudio[i] if (i< len(labels_fromAudio)) else labels_fromAudio[-1] for i in thisValidAudioFrames]
                except: import pdb;pdb.set_trace()

    # if the checks succeed, set final values and convert to proper formats
    thisValidAudioFrames = np.array(thisValidAudioFrames)
    thisAudioLabels = np.array(map(int, [phoneme_set_39[phoneme] for phoneme in labels_fromAudio]))
    thisValidLabels = np.array(map(int, [phoneme_set_39[phoneme] for phoneme in validLabels_fromAudio]))

    # now convert to proper format
    input_data_type = 'float32'
    thisMFCCs = set_type(thisMFCCs, input_data_type)

    label_data_type = 'int32'
    thisAudioLabels = set_type(thisAudioLabels, label_data_type)
    thisValidLabels = set_type(thisValidLabels, label_data_type)
    thisValidAudioFrames = set_type(thisValidAudioFrames, label_data_type)

    return thisMFCCs, thisValidLabels, thisAudioLabels, thisValidAudioFrames

def getWrongIndices(a,b):
    wrong = []
    assert len(a) == len(b)
    for i in range(len(a)):
        if a[i]  != b[i]: wrong.append(i)
    return wrong


def normalizeMFCC(MFCC, mean, std_dev):
    return (MFCC - mean) / std_dev

def speakerToBinary_perVideo(speakerDir, binaryDatabaseDir, mean, std_dev, test=False, overWrite=True):  # meanStdAudio is tuple of mean and std_dev of audio training data
    targetDir = binaryDatabaseDir
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    # write label and image to binary file, 1 label+image per row
    speakerName = os.path.basename(speakerDir)
    outputPath = targetDir + os.sep + speakerName + ".pkl"

    if os.path.exists(outputPath) and not overWrite:
        logger_combinedPrep.info("%s files have NOT been written; %s already exists", speakerName, outputPath)
        return []

    badDirsThisSpeaker =[]

    allvideosImages = []
    allvideosMFCCs = []
    allvideosValidLabels = []
    allvideosAudioLabels = []
    allvideosValidAudioFrames = []  #valid audio frames

    videoSequence = []

    print(speakerDir)
    dirs = directories(speakerDir)
    #import pdb;pdb.set_trace()
    if 'Lipspkr' in speakerDir:  #ugly hack because the dirs for combinedSR/dataToPkl were not sorted, just random. We need the same order to have same train/val/test videos
        nr = speakerDir[-1]
        dirOrder = unpickle('/home/matthijs/TCDTIMIT/lip'+str(nr)+'order.pkl')
        dirs = [speakerDir+os.sep+dirtje for dirtje in dirOrder]
    #import pdb;pdb.set_trace()

    for videoDir in dirs:  # rootDir is the speakerDir, below that are the videodirs
        print(videoDir)
        videoSequence.append(os.path.basename(videoDir))
        #logger_combinedPrep.info("    Extracting video: %s", os.path.basename(videoDir))
        thisMFCCs, thisValidLabels, thisAudioLabels, thisValidAudioFrames = audioDirToArrays(videoDir, nbMFCCs)
        thisMFCCs = normalizeMFCC(thisMFCCs, mean, std_dev)

        allvideosMFCCs.append(thisMFCCs)
        allvideosAudioLabels.append(thisAudioLabels)
        allvideosValidLabels.append(thisValidLabels)
        allvideosValidAudioFrames.append(thisValidAudioFrames)

    # now write python dict to a file
    logger_combinedPrep.info("the data file takes: %s bytes of memory", sys.getsizeof(allvideosImages))

    mfccs_test = list(allvideosMFCCs)
    audioLabels_test = list(allvideosAudioLabels)
    validLabels_test = list(allvideosValidLabels)
    validAudioFrames_test = list(allvideosValidAudioFrames)

    dtypeX = 'float32'
    dtypeY = 'int32'
    if isinstance(mfccs_test, list):                mfccs_test = set_type(mfccs_test, dtypeX);
    if isinstance(audioLabels_test, list):          audioLabels_test = set_type(audioLabels_test, dtypeY);
    if isinstance(validLabels_test, list):          validLabels_test = set_type(validLabels_test, dtypeY);
    if isinstance(validAudioFrames_test, list):     validAudioFrames_test = set_type(validAudioFrames_test, dtypeY);

    data = [mfccs_test, audioLabels_test, validLabels_test, validAudioFrames_test]
    output = open(outputPath, 'wb');
    cPickle.dump(data, output, 2);
    output.close()
    logger_combinedPrep.info("%s files have been written to: %s", speakerName, outputPath)
    return 0


def allSpeakersToBinary(databaseDir, binaryDatabaseDir, meanStdAudio, test=False, overWrite=True):
    mean = meanStdAudio[0]
    std_dev = meanStdAudio[1]

    rootDir = databaseDir
    dirList = []
    for dir in directories(rootDir):
        # logger_combinedPrep.info(dir)
        # logger_combinedPrep.info(relpath(rootDir, dir))
        # logger_combinedPrep.info(depth(relpath(rootDir, dir)))
        if depth(relpath(rootDir, dir)) == 0:
            dirList.append(dir)
    logger_combinedPrep.info("\n %s", [os.path.basename(directory) for directory in dirList])
    dirList = sorted(dirList)

    for speakerDir in tqdm(dirList, total=len(dirList)):
        logger_combinedPrep.info("\nExtracting files of speaker: %s", os.path.basename(speakerDir))
        speakerToBinary_perVideo(speakerDir, binaryDatabaseDir, mean, std_dev, test, overWrite=overWrite)

    return 0


if __name__ == "__main__":
    main()
