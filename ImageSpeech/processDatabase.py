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
    validTimes = [float(phoneme[0]) for phoneme in phonemes]
    
    validFrames = [int(round(validTime* framerate)) for validTime in validTimes]
    validFrames[-1]= len(videoROI)-1  #ugly fix because the database mat file ends too early
    logging.info("Total nb of valid frames in the video: \t %d", len(validFrames))

    validPhonemes = [phoneme[1] for phoneme in phonemes]
    logging.info("%s", '\t | '.join([str(validTime) for validTime in validTimes]))
    logging.info("\t %s", '\t | '.join([str(validFrame) for validFrame in validFrames]))
    logging.info("%s", '\t | '.join([str(validPhoneme) for validPhoneme in validPhonemes]))
    
    # get images corresponding to the valid frames
    validImages = [videoROI[validFrame] for validFrame in validFrames]      # store image  #TODO not correct frame?
    
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
            result = saveMatFile(videoPath, phonemes, storeDir)
            if result != 0:
                errorList.append(result)
            print("NUMBER OF ERRORS: ", nbErrors)
            print("-----------------------------------------------------------------------")

    return 0


# extract the images at the times of the phonemes from the videom giving them a nice name `videoName_timestamp_phoneme`. Crop approximate mouth region
def videoToImages (videoPath, phonemes, targetDir, targetSize='296:224', cropStartPixel='888:614'):
    print("Processing: " + videoPath)
    
    for i in range(len(phonemes)):
        extractionTimeFloat = phonemes[i][0]
        extractionTime = str(extractionTimeFloat).replace('.', '-')  # needs this format for ffmpeg
        
        phoneme = phonemes[i][1]
        
        videoName = os.path.basename(videoPath)
        videoName = os.path.splitext(videoName)[0]  # remove extension
        outputPath = ''.join([targetDir, os.sep, videoName, "_", extractionTime, "_", phoneme,
                              ".jpg"])  # eg vid1_00-00-01-135_sh.jpg
        
        # print(outputPath)
        # from https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
        if not os.path.exists(targetDir): os.makedirs(targetDir)
        command = ['ffmpeg',
                   '-ss', "00:00:" + extractionTimeFloat,
                   '-i', videoPath,
                   '-s', targetSize,
                   '-vf', "crop=" + targetSize + ":" + cropStartPixel,
                   '-frames:v', '1',
                   outputPath]
        
        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
    return 0

