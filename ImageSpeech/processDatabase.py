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

import dlib
from skimage import io
import cv2

from helpFunctions import *

# read all the times from the mlf file, split on line with new video
# Return:           a list, where each list element contains the text of one video, stored in another list, line by line

# create a list of lists. The higher-level list contains the block of one video, the 2nd-level list contains all the lines of that video
# MLFfile
#    | video1
#        | firstPhoneme
#        | secondPhoneme
#    | video2
#    etc        
#http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python#3277516

def readfile(filename):
    with open(filename, "r") as ins:
        array = [[]]
        video_index = 0
        for line in ins:
            line = line.strip('\n') # strip newlines
            if len(line)>1: # don't save the dots lines
                if ".rec" not in line:
                    array[video_index].append(line)
                else: #create new 2nd-level list, now store there
                    array.append([line])
                    video_index += 1

    return array[1:]


# outputs a list of times where the video should be converted to an image, with the corresponding phonemes spoken at those times
# videoPhonemeList:        list of lines from read file that cover the phonemes of one video,
#                              as well as  a list of the phonemes at those times (for keeping track of which image belongs to what)
# timeModifier:            extract image at beginning, middle, end of phoneme interval?
#                            (value between 0 and 1, 0 meaning at beginning)
def processVideoFile(videoPhonemeList, timeModifier = 1):
    videoPath =  str(videoPhonemeList[0]).replace('"','')
    videoPath =  videoPath.replace('rec','mp4')

    phonemes = []                               #list of tuples, 1= time. 2=phoneme

    for idx, line in enumerate(videoPhonemeList[1:]):           #skip the path line; then just three columns with tabs in between
        splittedLine = line.split()             #split on whitespaces
       
        phoneme = splittedLine[2]
        
        start = float(splittedLine[0])/10000000
        end = float(splittedLine[1])/10000000
        
        if (idx == 0 ):  #beginning or end = silence, take the best part  #TODO
            extractionTime = start
        elif (idx == len(videoPhonemeList[1:])-1):
            extractionTime = end
        else:
            extractionTime = start * (1 - timeModifier) + (timeModifier) * end
        extractionTime = "{0:.3f}".format(extractionTime)   #three decimals
        
        phonemes.append( (extractionTime, phoneme) )        # add the (time,phoneme) tuple

    return videoPath, phonemes

# store phonemes in a file with a name corresponding to the video they belong to
def writePhonemesToFile(videoName, phonemes, targetDir):
    phonemeFileName = videoName + "_PHN"

    thefile = open(targetDir + os.sep + phonemeFileName, 'w')
    for item in phonemes:
          thefile.write(' '.join(map(str,item)) + "\r\n")
    thefile.close()
    return 0

# create a new .mat file that contains only the frames we found a phoneme
# videoPath :    self-explanatory
# phonemes:        list of (phoneme, time) tuples
def saveMatFile(videoPath, phonemes, targetDir, framerate=29.97):        # frameRate = 29.97 for the TCDTimit database
    base = os.path.splitext(videoPath)[0]
    videoName = os.path.basename(videoPath)
    videoName = os.path.splitext(videoName)[0]  # remove extension
    videoROIfile = base + ".mat"                # a .mat file that contains a cell, that contains a row of matrices. Each matrix represents a mouth ROI for one video frame
    
    videoROI = scipy.io.loadmat(videoROIfile)
    videoROI = videoROI['ROIs'].tolist()
    videoROI = videoROI[0][0][0].tolist()       # videoROI is now a list that contains a matrix for every video frame
    logging.info("Total nb of frames in the video: \t\t\t %d",len(videoROI))

    # gather the used frame numbers
    validTimes, validFrames, validPhonemes = getValid(phonemes,framerate)
    logging.info("%s", '\t | '.join([str(validTime) for validTime in validTimes]))
    logging.info("\t %s", '\t | '.join([str(validFrame) for validFrame in validFrames]))
    logging.info("%s", '\t | '.join([str(validPhoneme) for validPhoneme in validPhonemes]))
    
    # get images corresponding to the valid frames
    validImages = [videoROI[validFrame] for validFrame in validFrames[1:-2]]      # store image  #TODO not correct frame?  #TODO ugly hack becasue not enough frames in the .mat file
    validImages.append(videoROI[-1])
    
    # prepare path
    outputPath = ''.join([targetDir, os.sep, videoName, "_validFrames.mat"])
    # store the data; name of file, and dictionary of the variables to save
    try:
        scipy.io.savemat(outputPath, {'validImages': validImages, 'validFrames': validFrames,'validPhonemes': validPhonemes})
        print("saved mat file ", videoName, "_validFrames.mat", " containing", len(validFrames), " images.")
        return 0
    except Exception, e:
        print("Couldn't do it: ", e)
        tb = traceback.format_exc()
        logging.warning(tb)
        #raw_input("Press Enter to continue...")
        pass
    return videoName #add one to error counter

# get valid times, phonemes, frame numbers
def getValid(phonemes, framerate):  # frameRate = 29.97 for the TCDTimit database
    import math
    validTimes = [float(phoneme[0]) for phoneme in phonemes]
    validFrames = [int(math.floor(validTime * framerate)) for validTime in validTimes]
    validPhonemes = [phoneme[1] for phoneme in phonemes]
    return validTimes, validFrames, validPhonemes

# process an MLF file containing video paths, and corresponding phoneme/time combinations; extract the corresponding frames from the .mat file and store it all in a new mat file.
# this new mat file contains three variables: 'validImages' and 'validPhonemes'. We can use these for network training.
def processMLF(MLFfile, storeDir):
    videos = readfile(MLFfile)
    nbErrors = 0
    errorlist = []
    for video in videos:
            videoPath, phonemes = processVideoFile(video) # phonemes: tuple of (time, phoneme)
            videoName = os.path.basename(videoPath)
            videoName = os.path.splitext(videoName)[0]  # remove extension
            print("Processing ", videoPath , " ...")
            logging.info("\t phonemes: %s", ':'.join([str(phoneme) for phoneme in phonemes]))
    
            #writePhonemesToFile(videoName, phonemes, storeDir)
            #videoToImages(videoPath, phonemes, storeDir, targetSize='296:224')
            result= saveMatFile(videoPath, phonemes, storeDir)
            
            if result != 0:
                errorList.append(result)
            print("NUMBER OF ERRORS: ", nbErrors)
            print("-----------------------------------------------------------------------")

    return 0


def extractFaces(sourceDir, storeDir, videoName):
    """
     detect lower half of face on every file, store them in storeDir
    """
    #storeDir / videoName / VideoName_frameNumber.jpg
    from os import listdir
    from os.path import isfile, join
    dirPath = sourceDir
    onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    onlyfiles.sort(key=tryint)  # sorts list in place
    #print(onlyfiles)
    if not os.path.exists(storeDir):
        os.makedirs(storeDir)
    for file in onlyfiles:
        if "face" in file: continue #skip face-extraced files
        filename, ext = os.path.splitext(file)
        videoName, frame = filename.split("_")
        filePath = ''.join([dirPath,os.sep,file]) #the file we're processing now
        storePath = ''.join([storeDir, os.sep, videoName, "_face_", str(frame), ".jpg"]) # the face will be saved here
        if os.path.exists(storePath):continue #skip existing 'face' files
        
        print("Exctracting face from: ", file)
        extractFace(filePath, storePath)
     
     
def removeInvalidFrames (phonemes, videoName, storeDir, framerate):
    """
    remove all frames not in validFrames
    """
    validTimes, validFrames, validPhonemes = getValid(phonemes,framerate)
    print(len(validFrames)," | ", validFrames)
    for frame in range(validFrames[0], validFrames[-1]):
        if frame not in validFrames:
            path = ''.join([storeDir, os.sep, videoName, "_", str(frame+1), ".jpg"]) #eg TCD-TIMIT/lipspeakers/LipSkr1/sa1/sa1_59.jpg
            silentremove(path)
    return 0


def extractAllFrames (videoPath, videoName, storeDir, framerate, targetSize='960x540'):
    """
    extract all frames from a video, and store them in storeDir
    """
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    
    if not os.path.exists(storeDir):  # skip already existing videos (it is assumed the exist if the directory exists)
        os.makedirs(storeDir)
        # eg vid1_. frame number and extension will be added by ffmpeg
        outputPath = ''.join([storeDir, os.sep, videoName, "_", ]) # eg .../sa1_3.jpg (frame and extension added by ffmpg)

        command = ['ffmpeg',
               '-i', videoPath,
               '-s', targetSize,
               outputPath + "%d.jpg"]  # add frame number and extension

        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
        return 0

    else:
        return 0 #files already exist?


def fixStoreDirName (topDirName, videoName, pathLine):
    """
    Fix the path of the root dir of all the newly generated files for this video.
    Gets base path from the MLF file; removes everything from 'Clips' on; adds storeDirName
    For example: file for lipspeaker will be '/media/matthijs/TOSHIBA EXT/TCDTIMIT/processed/lipspeakers/Lipspkr1'
    :param storeDir: the name of the root dir (which will be just under the 'TCDTIMIT' dir)
    :return:
    """
    storeDir = str(pathLine).replace('"', '')
    storeDir = storeDir.replace('rec', 'mp4')
    storeDir = storeDir.replace("TCDTIMIT", "TCDTIMIT/"+topDirName)
    storeDir, second = storeDir.split("Clips")
    if storeDir.endswith('/'):
        storeDir = storeDir[:-1]
        
    # now add the video Name
    storeDir = ''.join([storeDir, os.sep, videoName])
    return storeDir

  
  
    ### TESTING ###
print("###################################")
videos = readfile('./lipspeaker_labelfiles.mlf')
framerate = 29.97
topDir = "processed"
batchSize = 10 # number of videos per iteration

batchIndex = 0
running = 1


def extractFacesFunction(video):
    videoPath, phonemes = processVideoFile(video)
    videoName = os.path.splitext(os.path.basename(videoPath))[0]
    sourceDir = fixStoreDirName(topDir, videoName, video[0])
    storeDir = ''.join([sourceDir, os.sep, "faces"])
    print("getting data from: \t", sourceDir)
    print("saving faces to: \t", storeDir)
    extractFaces(sourceDir, storeDir, videoName)


while running:
    if batchIndex+batchSize > len(videos):
        print("Processing LAST BATCH of videos...")
        running = 0
        currentVideos = videos[batchIndex:] #till the end
    else:
        currentVideos = videos[batchIndex:batchIndex+batchSize]
    # 1. extract the frames
    for video in currentVideos:
        videoPath, phonemes = processVideoFile(video)
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        storeDir = fixStoreDirName(topDir, videoName, video[0]) #eg storeDir = '/media/matthijs/TOSHIBA EXT/TCDTIMIT/processed/lipspeakers/LipSpkr1'
        print("saving frames to: \t", storeDir)
        extractAllFrames(videoPath, videoName, storeDir, framerate, '1920x1080')
    sleepTime = 10+len(currentVideos)*2
    print("Sleepping for ",sleepTime, " seconds to allow files to be written to disk.")
    time.sleep(sleepTime) # wait till files have been written
    print("\tAll frames extracted.")
    print("----------------------------------")

    # 2. remove unneccessary frames
    for video in currentVideos:
        videoPath, phonemes = processVideoFile(video)
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        storeDir = fixStoreDirName(topDir, videoName, video[0])
        print("removing frames from: \t", storeDir)
        removeInvalidFrames(phonemes, videoName, storeDir, framerate)
    #time.sleep(len(videos))
    print("\tAll unnecessary frames removed.")
    print("----------------------------------")

    # 3. extract faces, store in subdir 'faces'
    for video in currentVideos:
        extractFacesFunction()
    print("\tAll faces extracted.")
    print("----------------------------------")
    
    # update the batchIndex
    batchIndex += batchSize

print("All done.")


def extractMouths (phonemes, videoPath, storeDir, framerate=29.97):
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    # 2. calculate needed frames from video labels
    # 3. throw away all non-needed frames
    # 4. compress
    # 5. extract face
    # 6. extract mouth
    if not os.path.exists(videoPath):
        print("This video does not exist:", videoPath)
        return -1
    
    videoName = os.path.basename(videoPath)
    videoName = os.path.splitext(videoName)[0]  # remove extension
    
    if os.path.exists(storeDir + os.sep + videoName):
        print("video already processed. Skipping...")
        return 0
    
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    # stored in 'storeDir/videoName/VideoName_frameNumber.jpg'
    extractAllFrames(videoPath, storeDir, framerate, '1920x1080')
    
    # this takes some time, extract for another video before running the next command
    
    # 2. calculate needed frames from video labels
    removeInvalidFrames(phonemes, videoName, storeDir, framerate)
    
    # 3. extract face from images
    # extractFaces(storeDir, videoName)




