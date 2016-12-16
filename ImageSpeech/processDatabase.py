# from http://stackoverflow.com/questions/10672578/extract-video-frames-in-python#10672679

# Goal: parametrized, automated version of
#       ffmpeg -i n.mp4 -ss 00:00:20 -s 160x120 -r 1 -f singlejpeg myframe.jpg
from __future__ import print_function

import os, sys
from PIL import Image
import subprocess
import pickle
import scipy.io
import numpy as np
import logging, sys
import traceback
import time
import concurrent.futures
import dlib
from skimage import io
import cv2

from helpFunctions import *

logging.basicConfig(stream=sys.stderr,
                    level=logging.DEBUG)  # http://stackoverflow.com/questions/6579496/using-print-statements-only-to-debug


# levels: debug, info, warning, error and critical.
# logging.debug('A debug message!')
# logging.info('We processed %d records', len(processed_records))

# read all the times from the mlf file, split on line with new video
# Return:           a list, where each list element contains the text of one video, stored in another list, line by line

# create a list of lists. The higher-level list contains the block of one video, the 2nd-level list contains all the lines of that video
# MLFfile
#    | video1
#        | firstPhoneme
#        | secondPhoneme
#    | video2
#    etc
# http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python#3277516

def readfile (filename):
    with open(filename, "r") as ins:
        array = [[]]
        video_index = 0
        for line in ins:
            line = line.strip('\n')  # strip newlines
            if len(line) > 1:  # don't save the dots lines
                if ".mp4" not in line:
                    array[video_index].append(line)
                else:  # create new 2nd-level list, now store there
                    array.append([line])
                    video_index += 1
    
    return array[1:]


# outputs a list of times where the video should be converted to an image, with the corresponding phonemes spoken at those times
# videoPhonemeList:        list of lines from read file that cover the phonemes of one video,
#                              as well as  a list of the phonemes at those times (for keeping track of which image belongs to what)
# timeModifier:            extract image at beginning, middle, end of phoneme interval?
#                            (value between 0 and 1, 0 meaning at beginning)
def processVideoFile (videoPhonemeList, timeModifier=0.5):
    videoPath = str(videoPhonemeList[0]).replace('"', '')
    videoPath = videoPath.replace('rec', 'mp4')
    
    phonemes = []  # list of tuples, 1= time. 2=phoneme
    
    for idx, line in enumerate(
            videoPhonemeList[1:]):  # skip the path line; then just three columns with tabs in between
        splittedLine = line.split()  # split on whitespaces
        
        phoneme = splittedLine[2]
        
        start = float(splittedLine[0]) / 10000000
        end = float(splittedLine[1]) / 10000000
        
        # if (idx == 0 ):  #beginning or end = silence, take the best part  #TODO
        #     extractionTime = start
        # elif (idx == len(videoPhonemeList[1:])-1):
        #     extractionTime = end
        # else:
        extractionTime = start * (1 - timeModifier) + (timeModifier) * end
        extractionTime = "{0:.3f}".format(extractionTime)  # three decimals
        
        phonemes.append((extractionTime, phoneme))  # add the (time,phoneme) tuple
    
    return videoPath, phonemes


# get valid times, phonemes, frame numbers
def getValid (phonemes, framerate):  # frameRate = 29.97 for the TCDTimit database
    import math
    # print("PHONEMES: ", phonemes)
    # validTimes = [float(phoneme[0]) for phoneme in phonemes]
    # print(validTimes)
    # validFrames = [int(math.floor(validTime * framerate)) for validTime in validTimes]
    # print(validFrames)
    # validPhonemes = [phoneme[1] for phoneme in phonemes]
    
    # take care of duplicates: loop through the phonemes, if two are same frame, only keep the first one
    seenFrames = set()
    doubleFrames = set()
    validFrames = []
    validPhonemes = []
    validTimes = []
    for phoneme in phonemes:
        time = float(phoneme[0])
        frame = int(math.floor(time * framerate))
        if frame not in seenFrames:
            validPhonemes.append(phoneme[1])
            validTimes.append(time)
            validFrames.append(frame)
            seenFrames.add(frame)
        else:
            print("frame ", frame, " already seen")
            doubleFrames.add(frame)
    # print(validFrames)
    # print(doubleFrames)
    return validTimes, validFrames, validPhonemes


# write file with phonemes and corresponding frame numbers. First column = frames. Second column = corresponding phonemes
def writePhonemesToFile (videoName, speakerName, phonemes, targetDir):
    validTimes, validFrames, validPhonemes = getValid(phonemes, 29.97)
    phonemeFile = ''.join([targetDir, os.sep, speakerName, "_", videoName, "_PHN.txt"])
    
    # add 1 to the validFrames to fix the ffmpeg issue (starts at 1 instead of 0)
    for i in range(0, len(validFrames)):
        validFrames[i] += 1
    
    # write to file
    thefile = open(phonemeFile, 'w')
    for i in range(len(validFrames) - 1):
        item = (validFrames[i], validPhonemes[i])
        thefile.write(' '.join(map(str, item)) + "\r\n")
    item = (validFrames[-1], validPhonemes[-1])
    thefile.write(' '.join(map(str, item)))
    thefile.close()
    
    matPath = targetDir + os.sep + "phonemeFrames.mat"
    sio.savemat(matPath, {'validFrames': np.array(validFrames), 'validPhonemes': np.array(validPhonemes)})
    
    return 0


def removeInvalidFrames (phonemes, videoName, storeDir, framerate):
    validTimes, validFrames, validPhonemes = getValid(phonemes, framerate)
    # print(len(validFrames)," | ", validFrames)
    for frame in range(validFrames[0], validFrames[-1]):
        # path: eg /media/TCD-TIMIT/lipspeakers/LipSkr1/sa1/sa1_59.jpg
        path = ''.join([storeDir, os.sep, videoName, "_", str(frame + 1), ".jpg"])
        if frame not in validFrames:
            silentremove(path)
    
    return 0


#####################################
########### Main Function  ##########
#####################################

### Executing ###
def processDatabase (MLFfile, storageLocation, nbThreads=2):
    print("###################################")
    videos = readfile(MLFfile)
    print("There are ", len(videos), " videos to be processed...")
    framerate = 29.97
    batchSize = nbThreads  # number of videos per iteration
    
    print("This program will process all video files specified in {mlf}. It will store the extracted faces and mouths in {storageLocation}. \n \
            The process might take a while (for the lipspeaker files ~3h, for the volunteer files ~10h)".format(
        mlf=MLFfile, storageLocation=storageLocation))
    if query_yes_no("Are you sure this is correct?", "no"):
        
        batchIndex = 0
        running = 1
        # multithread the operations
        executor = concurrent.futures.ThreadPoolExecutor(nbThreads)
        
        while running:
            if batchIndex + batchSize >= len(videos):
                print("Processing LAST BATCH of videos...")
                running = 0
                currentVideos = videos[batchIndex:]  # till the end
            else:
                currentVideos = videos[batchIndex:batchIndex + batchSize]
            
            # 1. extract the frames
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                if not os.path.exists(videoPath):
                    print("The file ", videoPath, " does not exist.")
                    logging.critical("Stopping...")
                    running = 0;
                    return -1
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                storeDir = fixStoreDirName(storageLocation, videoName, video[0])
                print("Extracting frames from ", videoPath, ", saving to: \t", storeDir)
                futures.append(
                    executor.submit(extractAllFrames, videoPath, videoName, storeDir, framerate, '1200:1000', '350:0'))
                # write phonemes and frame numbers to file
                # print("writing phonemes...")
                # speakerName = os.path.basename(os.path.dirname(storeDir))
                # futures.append(executor.submit(writePhonemesToFile,videoName, speakerName, phonemes, storeDir))
            concurrent.futures.wait(futures)
            
            print([future.result() for future in futures])
            nbVideosExtracted = sum([future.result() for future in futures])
            sleepTime = nbVideosExtracted * 5
            print("Sleeping for ", sleepTime, " seconds to allow files to be written to disk.")
            time.sleep(sleepTime)  # wait till files have been written
            
            print("\tAll frames extracted.")
            print("----------------------------------")
            
            # if query_yes_no("stop?", "yes"): break
            
            # 2. extract the phonemes
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                if not os.path.exists(videoPath):
                    print("The file ", videoPath, " does not exist.")
                    logging.critical("Stopping...")
                    running = 0;
                    return -1
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                storeDir = fixStoreDirName(storageLocation, videoName, video[0])
                print("Extracting phonemes from ", videoPath, ", saving to: \t", storeDir)
                # write phonemes and frame numbers to file
                speakerName = os.path.basename(os.path.dirname(storeDir))
                futures.append(executor.submit(writePhonemesToFile, videoName, speakerName, phonemes, storeDir))
            concurrent.futures.wait(futures)
            print("phonemes have been written")
            print("-----------------------------")
            
            # 2. remove unneccessary frames
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                videoDir = fixStoreDirName(storageLocation, videoName, video[0])
                print("removing invalid frames from ", storeDir)
                futures.append(executor.submit(deleteUnneededFiles, videoDir))
            concurrent.futures.wait(futures)
            print("\tAll unnecessary frames removed.")
            print("----------------------------------")
            nbRemoved = sum([future.result() for future in futures])
            sleepTime = nbRemoved * 0.01
            time.sleep(sleepTime)
            
            # 3. extract faces and mouths
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                sourceDir = fixStoreDirName(storageLocation, videoName, video[0])
                storeDir = sourceDir
                print("Extracting faces from ", sourceDir)
                # exectute. The third argument is the path to the dlib facial landmark predictor
                futures.append(executor.submit(extractFacesMouths, sourceDir, storeDir,
                                               "./shape_predictor_68_face_landmarks.dat"))  # storeDir = sourceDir
            concurrent.futures.wait(futures)
            print("\tAll faces and mouths have been extracted.")
            print("----------------------------------")
            
            # 4. convert to grayscale
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                sourceDir = fixStoreDirName(storageLocation, videoName, video[0])
                storeDir = sourceDir
                dirNames = ["faces", "mouths"]
                print("Converting to grayscale from: ", sourceDir)
                futures.append(executor.submit(convertToGrayScale, sourceDir, dirNames))
            concurrent.futures.wait(futures)
            print("\tAll faces and mouths have been converted to grayscale.")
            print("----------------------------------")
            
            # 5. resize mouth images, for convnet usage
            futures = []
            for video in currentVideos:
                videoPath, phonemes = processVideoFile(video)
                videoName = os.path.splitext(os.path.basename(videoPath))[0]
                storeDir = fixStoreDirName(storageLocation, videoName, video[
                    0])  # eg /media/matthijs/TOSHIBA EXT/TCDTIMIT/processed/lipspeakers/LipSpkr1/sa1
                rootDir = storeDir
                dirNames = ["mouths_gray", "faces_gray"]
                print("Resizing images from: ", sourceDir)
                futures.append(executor.submit(resizeImages, storeDir, dirNames, False, 120.0))
            concurrent.futures.wait(futures)
            print("\tAll mouths have been resized.")
            print("----------------------------------")
            
            print("#####################################")
            print("\t Batch Done")
            print("#####################################")
            
            # update the batchIndex
            batchIndex += batchSize
        
        print("All done.")
    else:
        print("Okay then, goodbye!")
    return 0
