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

# detect faces in all jpg's in sourceDir
# extract faces to "storeDir/faces", and mouths to "storeDir/mouths"
def extractFacesMouths (sourceDir, storeDir):
    import dlib
    storeFaceDir = storeDir + os.sep + "faces"
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    storeMouthsDir = storeDir + os.sep + "mouths"
    if not os.path.exists(storeMouthsDir):
        os.makedirs(storeMouthsDir)
    
    detector = dlib.get_frontal_face_detector()
    predictor_path = "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/ImageSpeech/mouthDetection/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    
    for f in glob.glob(os.path.join(sourceDir, "*.jpg")):
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
            print("no faces found for file: ", f)
        else:
            for i, d in enumerate(dets):
                # extract face, store in storeFacesDir
                left = d.left() * resizer
                right = d.right() * resizer
                top = d.top() * resizer
                bot = d.bottom() * resizer
                face_img = img[top:bot, left:right]
                io.imsave(facePath, face_img)  # don't write to disk if already exists
                
                # detect 68 keypoints
                shape = predictor(imgSmall, d)
                # Get the mouth landmarks.
                mx = shape.part(48).x * resizer
                mw = shape.part(54).x * resizer - mx
                my = shape.part(31).y * resizer
                mh = shape.part(57).y * resizer - my
                
                # scale them to get a better image
                widthScalar = 1.5
                heightScalar = 1
                mx = int(mx - (widthScalar - 1) / 2.0 * mw)
                # my = int(my - (heightScalar - 1)/2.0*mh) #not need,d we already have enough nose
                mw = int(mw * widthScalar)
                mh = int(mh * widthScalar)
                
                mouth_img = img[my:my + mh, mx:mx + mw]
                io.imsave(mouthPath, mouth_img)


#############################
########  1st GEN ###########
#############################

# store phonemes in a file with a name corresponding to the video they belong to
def writePhonemesToFile (videoName, phonemes, targetDir):
    phonemeFileName = videoName + "_PHN"
    
    thefile = open(targetDir + os.sep + phonemeFileName, 'w')
    for item in phonemes:
        thefile.write(' '.join(map(str, item)) + "\r\n")
    thefile.close()
    return 0


# create a new .mat file that contains only the frames we found a phoneme
# videoPath :    self-explanatory
# phonemes:        list of (phoneme, time) tuples
def saveMatFile (videoPath, phonemes, targetDir, framerate=29.97):  # frameRate = 29.97 for the TCDTimit database
    base = os.path.splitext(videoPath)[0]
    videoName = os.path.basename(videoPath)
    videoName = os.path.splitext(videoName)[0]  # remove extension
    videoROIfile = base + ".mat"  # a .mat file that contains a cell, that contains a row of matrices. Each matrix represents a mouth ROI for one video frame
    
    videoROI = scipy.io.loadmat(videoROIfile)
    videoROI = videoROI['ROIs'].tolist()
    videoROI = videoROI[0][0][0].tolist()  # videoROI is now a list that contains a matrix for every video frame
    logging.info("Total nb of frames in the video: \t\t\t %d", len(videoROI))
    
    # gather the used frame numbers
    validTimes, validFrames, validPhonemes = getValid(phonemes, framerate)
    logging.info("%s", '\t | '.join([str(validTime) for validTime in validTimes]))
    logging.info("\t %s", '\t | '.join([str(validFrame) for validFrame in validFrames]))
    logging.info("%s", '\t | '.join([str(validPhoneme) for validPhoneme in validPhonemes]))
    
    # get images corresponding to the valid frames
    validImages = [videoROI[validFrame] for validFrame in validFrames[
                                                          1:-2]]  # store image  #TODO not correct frame?  #TODO ugly hack becasue not enough frames in the .mat file
    validImages.append(videoROI[-1])
    
    # prepare path
    outputPath = ''.join([targetDir, os.sep, videoName, "_validFrames.mat"])
    # store the data; name of file, and dictionary of the variables to save
    try:
        scipy.io.savemat(outputPath,
                         {'validImages': validImages, 'validFrames': validFrames, 'validPhonemes': validPhonemes})
        print("saved mat file ", videoName, "_validFrames.mat", " containing", len(validFrames), " images.")
        return 0
    except Exception, e:
        print("Couldn't do it: ", e)
        tb = traceback.format_exc()
        logging.warning(tb)
        # raw_input("Press Enter to continue...")
        pass
    return videoName  # add one to error counter
