#!/usr/bin/python

#### help functions
from __future__ import print_function

# remove without complaining
import os, errno
import subprocess

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy as np
import scipy.io as sio

## http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input#3041990
# query_yes_no("Is cabbage yummier than cauliflower?", None)
# Is cabbage yummier than cauliflower? [y/n] [ENTER]
# Please respond with 'yes' or 'no' (or 'y' or 'n').
# Is cabbage yummier than cauliflower? [y/n] y
# >>> True
def query_yes_no (question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
            
def silentremove (filename):
    try:
        os.remove(filename)
    except OSError as e:  # name the Exception `e`
        #print("Failed with:", e.strerror)  # look what it says
        pass



# sort based on filenames, numerically instead of lexicographically
def tryint (s):
    t = os.path.splitext(s)[0]
    try:
        return int(t)
    except:
        try:
            u = t.split("_")[1]
            return int(u)
        except:
            return t
                       
# extract faces out of an image using dlib
import sys, os
import dlib
from skimage import io
import cv2


def resize_image (filePath, filePathResized, width=360.0):
    im = cv2.imread(filePath)
    r = width / im.shape[1]
    dim = (int(width), int(im.shape[0] * r))
    resized = cv2.resize(im, dim)
    cv2.imwrite(filePathResized, resized)


def resizeImages (dirPath, width=640.0):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    onlyfiles.sort(key=tryint)
    for file in onlyfiles:
        filename, ext = os.path.splitext(file)
        filePath = ''.join([dirPath, os.sep, file])  # the file we're processing now
        resize_image(filePath, filePath, width)
    return 0


def fixStoreDirName (storageLocation, videoName, pathLine):
    """
    Fix the path of the root dir of all the newly generated files for this video.
    Gets base path from the MLF file; removes everything from 'Clips' on; adds storeDirName
    For example: file for lipspeaker will be '/media/matthijs/TOSHIBA EXT/TCDTIMIT/processed/lipspeakers/Lipspkr1'
    :param storeDir: the name of the root dir (which will be just under the 'TCDTIMIT' dir)
    :return:
    """
    storeDir = str(pathLine).replace('"', '')
    storeDir = storeDir.replace('.rec', '.mp4')
    print(storeDir)
    oldStoragePath, relPath = storeDir.split("TCDTIMIT/") #/home/matthijs/TCDTIMIT/volunteers/...
    storeDir = ''.join([storageLocation, os.sep, relPath])
    storeDir, second = storeDir.split("Clips")
    if storeDir.endswith('/'):
        storeDir = storeDir[:-1]
    
    # now add the video Name
    storeDir = ''.join([storeDir, os.sep, videoName])
    return storeDir



def extractAllFrames (videoPath, videoName, storeDir, framerate, targetSize='640:640', cropStartPixel='640:300'):
    """
    extract all frames from a video, and store them in storeDir
    """
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    if not os.path.exists(storeDir):  # skip already existing videos (it is assumed the exist if the directory exists)
        os.makedirs(storeDir)
        # eg vid1_. frame number and extension will be added by ffmpeg
        outputPath = ''.join([storeDir, os.sep, videoName, "_", ])  # eg .../sa1_3.jpg (frame and extension added by ffmpg)
        
        command = ['ffmpeg',
                   '-i', videoPath,
                   '-s', targetSize,
                   '-vf', "crop=" + targetSize + ":" + cropStartPixel,
                   outputPath + "%d.jpg"]  # add frame number and extension
        
        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
        return 1
    
    else:
        return 0  # files already exist?

# write file with phonemes and corresponding frame numbers
def writePhonemesToFile (videoName, speakerName, phonemes, targetDir):
    # print(phonemes)
    validTimes, validFrames, validPhonemes = getValid(phonemes, 29.97)
    phonemeFile = ''.join([targetDir, os.sep, speakerName, "_", videoName, "_PHN.txt"])

    # add 1 to the validFrames to fix the ffmpeg issue (starts at 1 instead of 0)
    for i in range(0, len(validFrames)):
        validFrames[i] += 1

    # write to file
    thefile = open(phonemeFile, 'w')
    writeTuples = (validFrames, validPhonemes)
    for item in writeTuples:
        thefile.write(' '.join(map(str, item)) + "\r\n")
    thefile.close()

    sio.savemat('phonemeFrames.mat', {'validFrames': np.array(validFrames), 'validPhonemes': np.array(validPhonemes)})
    return 0


# detect faces in all jpg's in sourceDir
# extract faces to "storeDir/faces", and mouths to "storeDir/mouths"
def extractFacesMouths (sourceDir, storeDir, predictor_path):
    import dlib
    storeFaceDir = storeDir + os.sep + "faces"
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    storeMouthsDir = storeDir + os.sep + "mouths"
    if not os.path.exists(storeMouthsDir):
        os.makedirs(storeMouthsDir)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    for f in glob.glob(os.path.join(sourceDir, "*.jpg")):
        dets = []
        fname = os.path.splitext(os.path.basename(f))[0]
        facePath = storeFaceDir + os.sep + fname + "_face.jpg"
        mouthPath = storeMouthsDir + os.sep + fname + "_mouth.jpg"
        img = io.imread(f)
        
        # don't reprocess images that already have a face extracted, as we've already done this before!
        if os.path.exists(facePath): continue
        
        # detect face, then keypoints. Store face and mouth
        resizer = 4
        height, width = img.shape[:2]
        imgSmall = cv2.resize(img, (int(width / resizer), int(height / resizer)),
                              interpolation=cv2.INTER_AREA)  # linear for zooming, inter_area for shrinking
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
        dets = detector(imgSmall, 1)  # detect face, don't upsample
        
        if len(dets) == 0:
            print("no faces found for file: ", f, "; using full res image...")
            dets = []
            resizer = 1
            height, width = img.shape[:2]
            imgSmall = cv2.resize(img, (int(width / resizer), int(height / resizer)),
                                  interpolation=cv2.INTER_AREA)  # linear for zooming, inter_area for shrinking
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
            dets = detector(imgSmall, 1)  # detect face, don't upsample
            if len(dets) == 0:
                print("still no faces found. Skipping...")
            else:
                print("now we found a face.")
        
        for i, d in enumerate(dets):
            # extract face, store in storeFacesDir
            left = d.left() * resizer
            right = d.right() * resizer
            top = d.top() * resizer
            bot = d.bottom() * resizer
            # go no further than img borders
            if (left < 0):      left = 0
            if (right > width): right = width
            if (top < 0):       top = 0
            if (bot > height):  bot = height
            face_img = img[top:bot, left:right]
            io.imsave(facePath, face_img)  # don't write to disk if already exists
            
            # detect 68 keypoints
            shape = predictor(imgSmall, d)
            # Get the mouth landmarks.
            mx = shape.part(48).x * resizer
            mw = shape.part(54).x * resizer - mx
            my = shape.part(31).y * resizer
            mh = shape.part(57).y * resizer - my
            # go no further than img borders
            if (mx < 0):       mx = 0
            if (mw > width):   mw = width
            if (my < 0):       my = 0
            if (mh > height):  mh = height
            
            # scale them to get a better image
            widthScalar = 1.5
            heightScalar = 1
            mx = int(mx - (widthScalar - 1) / 2.0 * mw)
            # my = int(my - (heightScalar - 1)/2.0*mh) #not need,d we already have enough nose
            mw = int(mw * widthScalar)
            mh = int(mh * widthScalar)
            
            mouth_img = img[my:my + mh, mx:mx + mw]
            io.imsave(mouthPath, mouth_img)
