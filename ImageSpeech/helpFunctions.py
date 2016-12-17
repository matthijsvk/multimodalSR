#!/usr/bin/python

#### help functions
from __future__ import print_function

# remove without complaining
import os, errno
import subprocess
import getopt
import zipfile, os.path
import concurrent.futures
import threading

import shutil

import sys
import dlib
import glob
import numpy as np
import scipy.io as sio
import sys
import dlib
# import cv2

import time
from skimage.color import rgb2gray
from skimage import io
from skimage import data
from skimage.transform import resize

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
        # print("Failed with:", e.strerror)  # look what it says
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
    if not "TCDTIMIT/" in storeDir: raise Exception("You have to create a 'TCDTIMIT' top level directory!!"); sys.exit(
        -1)
    oldStoragePath, relPath = storeDir.split("TCDTIMIT/")  # /home/data/TCDTIMIT/volunteers/...
    storeDir = ''.join([storageLocation, os.sep, relPath])
    storeDir, second = storeDir.split("Clips")
    if storeDir.endswith('/'):
        storeDir = storeDir[:-1]
    
    # now add the video Name
    storeDir = ''.join([storeDir, os.sep, videoName])
    return storeDir


# delete unneeded files,  based on a phoneme file
def deleteUnneededFiles (videoDir):
    # read correct frames: firs column of text file
    parentName = os.path.basename(os.path.dirname(videoDir))
    dirName = os.path.basename(videoDir)
    validFrames = []
    with open(videoDir + os.sep + parentName + "_" + dirName + "_PHN.txt") as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                validFrames.append(parts[0])  # print column 2
    
    # walk through the files, if a file doesn't contain '_validFrame', then remove it.
    from os import listdir
    from os.path import isfile, join
    nbRemoved = 0
    for root, dirs, files in os.walk(videoDir):
        files.sort(key=tryint)
        for f in files:
            for validFrame in validFrames:
                remove = 1
                # only jpg
                name, ext = os.path.splitext(f)
                if ext != ".jpg": remove = 0; continue
                # must contain some validFrame
                fname = os.path.splitext(f)[0]
                fnumber = fname.split("_")[1]
                if validFrame == fnumber:
                    # print(f + " has frame number: ", fnumber, " and won't be removed")
                    remove = 0
                    break
            if remove == 1:
                filePath = os.path.join(root, f)
                # print(filePath + " will be removed")
                os.remove(filePath)
                nbRemoved += 1
    # print(validFrames)
    return nbRemoved


def extractAllFrames (videoPath, videoName, storeDir, framerate, targetSize, cropStartPixel):
    """
    extract all frames from a video, and store them in storeDir
    """
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    if not os.path.exists(storeDir):  # skip already existing videos (it is assumed the exist if the directory exists)
        os.makedirs(storeDir)
        # eg vid1_. frame number and extension will be added by ffmpeg
        outputPath = ''.join(
                [storeDir, os.sep, videoName, "_", ])  # eg .../sa1_3.jpg (frame and extension added by ffmpg)
        
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


# detect faces in all jpg's in sourceDir
# extract faces to "storeDir/faces", and mouths to "storeDir/mouths"
def extractFacesMouths (sourceDir, storeDir, predictor_path):
    if not os.path.exists(predictor_path):
        print('Landmark predictor not found!')
        sys.exit(1)
        
    storeFaceDir = storeDir + os.sep + "faces"
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    storeMouthsDir = storeDir + os.sep + "mouths"
    if not os.path.exists(storeMouthsDir):
        os.makedirs(storeMouthsDir)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    print("extrcati face")
    from os import listdir
    from os.path import isfile, join
    nbRemoved = 0
    
    print(sourceDir)
    
    for f in glob.glob(os.path.join(sourceDir, "*.jpg")):
                dets = []
                fname, ext = os.path.splitext(os.path.basename(f))
                print(f)
                if ext == ".jpg":
                    print(f)
                    facePath = storeFaceDir + os.sep + fname + "_face.jpg"
                    mouthPath = storeMouthsDir + os.sep + fname + "_mouth.jpg"
                    
                    if os.path.exists(facePath):
                        print(facePath, " already exists")
                        continue
                    
                    img = io.imread(f)
                    
                    # detect face, then keypoints. Store face and mouth
                    resizer = 4
                    
                    height, width = img.shape[:2]
                    # im_resized = skimage.transform.resize(im, (int(width / resizer), int(height / resizer)))
                    # imgSmall= skimage.color.rgb2gray(img_resized)
                    # io.imshow(imgSmall)
                    # io.show()
                    
                    imgSmall = cv2.resize(img, (int(width / resizer), int(height / resizer)),interpolation = cv2.INTER_AREA)  # linear for zooming, inter_area for shrinking
                    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
                    dets = detector(imgSmall, 1)  # detect face, don't upsample
                    
                    if len(dets) == 0:
                        print("looking on full-res image...")
                        resizer = 1
                        height, width = img.shape[:2]
                        # im_resized = skimage.transform.resize(im, (int(width / resizer), int(height / resizer)))
                        # imgSmall= skimage.color.rgb2gray(img_resized)
                        # io.imshow(imgSmall)
                        # io.show()
        
                        imgSmall = cv2.resize(img, (int(width / resizer), int(height / resizer)),interpolation=cv2.INTER_AREA)  # linear for zooming, inter_area for shrinking
                        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
                        
                        dets = detector(imgSmall, 1)  # detect face, don't upsample
                        if len(dets) == 0:
                            print("still no faces found. Using previous face coordinates...")
                            if 'top' in locals():
                                face_img = img[top:bot, left:right]
                                io.imsave(facePath, face_img)
                                mouth_img = img[my:my + mh, mx:mx + mw]
                                io.imsave(mouthPath, mouth_img)
                                continue
                            else:
                                print("top not in locals. ERROR")
                            continue
                    
                    d = dets[0]
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

def resize_image (filePath, filePathResized, keepAR=True, width=120.0):
    
    im = io.imread(filePath)
    if keepAR:
        r = width / im.shape[1]
        dim = (int(im.shape[0] * r), int(width))
        im_resized = resize(im, dim)
    else:
        im_resized = resize(im, (120, 120))
    io.imsave(filePathResized, im_resized)


def resizeImages (rootDir, dirNames, keepAR=True, width=640.0):
    from os import listdir
    from os.path import isfile, join
    for dirName in dirNames:
        dirPath = rootDir + os.sep + dirName
        targetDirPath = rootDir + os.sep + dirName + "_" + str(int(width))
        if not os.path.exists(targetDirPath):
            os.makedirs(targetDirPath)
        
        # loop through the files
        onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
        onlyfiles.sort(key=tryint)
        for file in onlyfiles:
            # print("processing ", file)
            filename, ext = os.path.splitext(file)
            filePath = ''.join([dirPath, os.sep, file])  # the file we're processing now
            targetPath = ''.join([targetDirPath, os.sep, file])
            # print("resizing ", filePath, " to ", targetPath)
            if not os.path.exists(targetPath):
                resize_image(filePath, targetPath, keepAR, width)
    return 0


# convert all images in folders specified in the list 'dirNames', that are found under the rootDir, to grayscale.
def convertToGrayScale (rootDir, dirNames):
    nbConverted = 0
    for root, dirs, files in os.walk(rootDir):
        files.sort(key=tryint)
        for file in files:
            parentDir = os.path.basename(root)
            fname = os.path.splitext(file)[0]  # no path, no extension. only filename
            if parentDir in dirNames:
                # convert all images in here to grayscale, store to dirName_gray
                newDirPath = ''.join([os.path.dirname(root), os.sep, parentDir + "_gray"])
                newFilePath = ''.join([newDirPath, os.sep, fname + "_gray.jpg"])
                if not os.path.exists(newDirPath):
                    os.makedirs(newDirPath)
                if not os.path.exists(newFilePath):
                    # read in grayscale, write to new path
                    # with OpenCV: weird results (gray image larger than color ?!?)
                    # img = cv2.imread(root+os.sep+file, 0)
                    # cv2.imwrite(newFilePath, img)
                    
                    from skimage.color import rgb2gray
                    from skimage import io
                    img_gray = rgb2gray(io.imread(root + os.sep + file))
                    io.imsave(newFilePath, img_gray)  # don't write to disk if already exists
                    nbConverted += 1
    
    # print(nbConverted, " files have been converted to Grayscale")
    return 0