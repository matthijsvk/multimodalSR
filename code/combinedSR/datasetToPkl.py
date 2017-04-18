import math
import os
from tqdm import tqdm
import timeit;
program_start_time = timeit.default_timer()
import random
random.seed(int(timeit.default_timer()))

# label files are in audio frames @16kHz, not in seconds
from PIL import Image
from phoneme_set import phoneme_set_39, classToPhoneme39
import math
import cPickle

import logging, formatting
logger_combinedPrep = logging.getLogger('prepcombined')
logger_combinedPrep.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger_combinedPrep.addHandler(ch)


from preprocessWavs import *
from general_tools import *

nbMFCCs = 39

def main():

    root = os.path.expanduser("~/TCDTIMIT/combinedSR/")  # ( keep the trailing slash)

    dataset = "TCDTIMIT"
    databaseDir = root + dataset + "/database"
    outputDir = root + dataset + "/binary"

    meanStdAudio = unpickle("./database_averages/TCDTIMITMeanStd.pkl")
    allSpeakersToBinary(databaseDir, outputDir, meanStdAudio)


def dirToArrays(dir, nbMFCCs=39, verbose=False):
    bad = []



    thisImages = []
    thisMFCCs = []
    thisValidAudioFrames = []  # these are the audio valid frames of this video

    labels_fromAudio = []
    imageValidFrames_fromAudio = []

    imagePaths = []
    for root, dirs, files in os.walk(dir):  # files in videoDir, should only be .jpg, .wav and .phn (and .vphn)
        for file in files:
            name, extension = os.path.splitext(file)
            if extension == ".jpg":
                imagePath = ''.join([root, os.sep, file])
                imagePaths.append(imagePath)

                # do the processing after all the files have been found, because then we can sort on frame number

            if extension == ".wav":
                # Get MFCC of the WAV
                wav_path = ''.join([root, os.sep, file])
                X_val, total_frames = create_mfcc('DUMMY', wav_path, nbMFCCs)
                total_frames = int(total_frames)
                thisMFCCs.append(X_val)

                # find the corresponding phoneme file, get the phoneme classes
                phn_path = ''.join([root, os.sep, name, ".phn"])
                if not os.path.exists(phn_path):
                    raise IOError("Phoneme files ", phn_path, " not found...")
                    import pdb;
                    pdb.set_trace()

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
                    imageValidFrames_fromAudio.append(frame + 1)

                    if verbose:
                        logger.debug('%s', (total_frames / float(total_duration)))
                        logger.debug('TIME  start: %s end: %s, phoneme: %s, class: %s', start_time, end_time,
                                     phoneme, phoneme_num)
                        logger.debug('FRAME start: %s end: %s, phoneme: %s, class: %s', start_ind, end_ind,
                                     phoneme, phoneme_num)
                fr.close()

                # save only the valid labels to this list
                try:validLabels_fromAudio = [labels_fromAudio[i] if (i< len(labels_fromAudio)) else labels_fromAudio[-1] for i in thisValidAudioFrames]
                except: import pdb;pdb.set_trace()

    imageValidFrames_fromImages = []  # just to check sanity, make sure audio and video correspond
    labels_fromImages = []  # use both to do sanity check
    imagePaths = sort_nicely(imagePaths)
    for imagePath in imagePaths:
        name = os.path.splitext(os.path.basename(imagePath))[0]
        try:videoName, frame, phoneme = name.split("_")
        except:
            print("Not a proper image, removing... : ", imagePath)
            os.remove(imagePath)
            continue

        # add the image, flattened as numpy array
        thisImages.append(np.array(Image.open(imagePath), dtype=np.uint8).flatten())
        # add the phoneme, converted to its class number
        labels_fromImages.append(phoneme)
        # save the imageFrame, to check later
        imageValidFrames_fromImages.append(int(frame))

    # do the sanity check on image frames and labels
    import pdb
    try:
        assert imageValidFrames_fromImages == imageValidFrames_fromImages
    # except:
    #     logger_combinedPrep.error(getWrongIndices(imageValidFrames_fromImages, imageValidFrames_fromImages));   pdb.set_trace()
    # # and on the labels
        assert labels_fromImages == validLabels_fromAudio
    except:
        # try: logger_combinedPrep.error(getWrongIndices(validLabels_fromAudio, labels_fromImages)); pdb.set_trace()
        # except:
        bad = [dir]; #pdb.set_trace()


    # if the checks succeed, set final values and convert to proper formats
    thisValidAudioFrames = np.array(thisValidAudioFrames)
    thisAudioLabels = np.array(map(int, [phoneme_set_39[phoneme] for phoneme in labels_fromAudio]))
    thisValidLabels = np.array(map(int, [phoneme_set_39[phoneme] for phoneme in validLabels_fromAudio]))

    # now convert to proper format
    input_data_type = 'float32'
    thisImages = set_type(thisImages, input_data_type)
    thisMFCCs = set_type(thisMFCCs, input_data_type)

    label_data_type = 'int32'
    thisAudioLabels = set_type(thisAudioLabels, label_data_type)
    thisValidLabels = set_type(thisValidLabels, label_data_type)
    thisValidAudioFrames = set_type(thisValidAudioFrames, label_data_type)


    return thisImages, thisMFCCs, thisValidLabels, thisAudioLabels, thisValidAudioFrames, bad

def getWrongIndices(a,b):
    wrong = []
    assert len(a) == len(b)
    for i in range(len(a)):
        if a[i]  != b[i]: wrong.append(i)
    return wrong


def normalizeMFCC(MFCC, mean, std_dev):
    return (MFCC - mean) / std_dev

def normalizeImages(X, verbose=False):
    # rescale to interval [-1,1], cast to float32 for GPU use
    X = np.multiply(2. / 255., X, dtype='float32')
    X = np.subtract(X, 1., dtype='float32');

    if verbose:  logger_combinedPrep.info("Train: %s %s", X.shape, X[0][0].dtype)

    # reshape to get one image per row
    X = np.reshape(X, (-1, 1, 120, 120))

    # cast to correct datatype, just to be sure. Everything needs to be float32 for GPU processing
    dtypeX = 'float32'
    X = X.astype(dtypeX);

    return X

def speakerToBinary_perVideo(speakerDir, binaryDatabaseDir, mean, std_dev):  # meanStdAudio is tuple of mean and std_dev of audio training data
    targetDir = binaryDatabaseDir
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    badDirsThisSpeaker =[]

    allvideosImages = []
    allvideosMFCCs = []
    allvideosValidLabels = []
    allvideosAudioLabels = []
    allvideosValidAudioFrames = []  #valid audio frames

    for videoDir in directories(speakerDir):  # rootDir is the speakerDir, below that are the videodirs
        #logger_combinedPrep.info("    Extracting video: %s", os.path.basename(videoDir))
        thisImages, thisMFCCs, thisValidLabels, thisAudioLabels, thisValidAudioFrames, badDirs= dirToArrays(videoDir, nbMFCCs)
        thisMFCCs = normalizeMFCC(thisMFCCs, mean, std_dev)
        thisImages = normalizeImages(thisImages)

        allvideosMFCCs.append(thisMFCCs)
        allvideosImages.append(thisImages)
        allvideosAudioLabels.append(thisAudioLabels)
        allvideosValidLabels.append(thisValidLabels)
        allvideosValidAudioFrames.append(thisValidAudioFrames)

        badDirsThisSpeaker = badDirsThisSpeaker + badDirs

    # write label and image to binary file, 1 label+image per row
    speakerName = os.path.basename(speakerDir)
    outputPath = targetDir + os.sep + speakerName + ".pkl"

    # now write python dict to a file
    logger_combinedPrep.info("the data file takes: %s bytes of memory", sys.getsizeof(allvideosImages))
    mydict = {'images': allvideosImages, 'mfccs': allvideosMFCCs, 'audioLabels': allvideosAudioLabels, 'validLabels': allvideosValidLabels, 'validAudioFrames': allvideosValidAudioFrames}
    output = open(outputPath, 'wb'); cPickle.dump(mydict, output, 2); output.close()

    logger_combinedPrep.info("%s files have been written to: %s", speakerName, outputPath)
    return badDirsThisSpeaker


def allSpeakersToBinary(databaseDir, binaryDatabaseDir, meanStdAudio):
    mean = meanStdAudio[0]
    std_dev = meanStdAudio[1]

    badDirectories = []

    rootDir = databaseDir
    dirList = []
    for dir in directories(rootDir):
        # logger_combinedPrep.info(dir)
        # logger_combinedPrep.info(relpath(rootDir, dir))
        # logger_combinedPrep.info(depth(relpath(rootDir, dir)))
        if depth(relpath(rootDir, dir)) == 1:
            dirList.append(dir)
    logger_combinedPrep.info("\n %s", [os.path.basename(directory) for directory in dirList])
    for speakerDir in tqdm(dirList, total=len(dirList)):
        logger_combinedPrep.info("\nExtracting files of speaker: %s", os.path.basename(speakerDir))
        badDirsSpeaker = speakerToBinary_perVideo(speakerDir, binaryDatabaseDir, mean, std_dev)
        badDirectories.append(badDirsSpeaker)

    print(badDirectories)
    pdb.set_trace()
    return 0


if __name__ == "__main__":
    main()
