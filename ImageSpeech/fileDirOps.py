# this file contains different operations on files and directories:
#   1. fixNames: files generated with old functions give mouth images, stored as 'videoName_faces_frameNb.jpg'
#                Transform this to the format 'videoName_frameNb_faces.jpg'
#
import sys
import getopt
import zipfile, os.path
import concurrent.futures
import threading

import os, errno
import subprocess
import shutil

from helpFunctions import *

# fix names so we can use them for training
def fixNames(rootDir):
    nbRenames = 0
    # step 1: Change names so that the 'faces' string is at the end of the filename, and it looks like: 'videoName_frameNb_faces.jpg'
    for root, dirs, files in os.walk(rootDir):
        files.sort(key=tryint)
        for file in files:
            if "face" in file: #sanity check
                filePath = os.path.join(root, file)
                parentDir = os.path.dirname(root)
                fname =  os.path.splitext(os.path.basename(file))[0]
                videoName, facestr, frame = fname.split("_")
                if facestr == "face": #if faces in the middle, swap frameNb and facestr
                    fileNew = ''.join([videoName, "_", frame, "_", facestr, ".jpg"])
                    fileNewPath = ''.join([root, fileNew])
                    fileNewPath = os.path.join(root, fileNew)
                    print(filePath+ "\t -> \t"+ fileNewPath)
                    #os.rename(filePath, fileNewPath)
                    nbRenames += 1
                    
    # Step 2: names are in proper format, now move to mouths folder (because the images contain mouths, not faces)
    nbMoves = 0
    for root, dirs, files in os.walk(rootDir):
        files.sort(key=tryint)
        for file in files:
            
            if "face" in file: #sanity check
                filePath = os.path.join(root, file)
                parentDir = os.path.dirname(root)
                fname = os.path.splitext(os.path.basename(file))[0]
                videoName, frame, facestr = fname.split("_")
                fileNew = ''.join([videoName, "_", frame, "_mouth.jpg"]) #replace 'face' with 'mouth'
                mouthsDir = ''.join([parentDir,os.sep,"mouths"])
                if not os.path.exists(mouthsDir):
                    os.makedirs(mouthsDir)
                fileNewPath = ''.join([mouthsDir,os.sep, fileNew])
                print(filePath + "\t -> \t" + fileNewPath)
                #os.rename(filePath, fileNewPath)
                nbMoves += 1
                 
    print(nbRenames, " files have been renamed.")
    print(nbMoves, "files have been moved.")
    return 0
# root = "/home/matthijs/TCDTIMIT/processed2"
# fixNames(root)

from shutil import copyfile
def frameToTiming(rootDir):
    nbCopies = 0
    for root, dirs, files in os.walk(rootDir):
        files.sort(key=tryint)
        for file in files:
            if "mouth" in file:
                filePath = os.path.join(root, file)
                parentDir = os.path.dirname(root)
                fname = os.path.splitext(os.path.basename(file))[0]
                videoName, frame, facestr = fname.split("_")
                timing = '%.3f' % float(int(frame)/29.97)
                timing = str(timing).replace('.', '-')
                fileNew = ''.join([videoName, "_", timing, "_mouth.jpg"]) #replace 'face' with 'mouth'
                mouthsDir = ''.join([parentDir,os.sep,"mouthsTiming"])
                if not os.path.exists(mouthsDir):
                    os.makedirs(mouthsDir)
                fileNewPath = ''.join([mouthsDir,os.sep, fileNew])
                print(filePath + "\t -> \t" + fileNewPath)
                #copyfile(filePath, fileNewPath)
                nbCopies += 1
    print(nbCopies, " files have been renamed")
    return 0

# root = "/home/matthijs/TCDTIMIT/processed2/lipspeakers/Lipspkr1/sa1"
# frameToTiming(root)

# remove all specified directories and their contents
# a rootdir, and a list of dirnames to be removed
# THIS FUNCTION deletes all specified directories AND their contents !!!
# Be careful!
def deleteDirs(rootDir, names):
    dirList= []
    for root, dirs, files in os.walk(rootDir):
        for dirname in dirs:
            for name in names:
                if name in dirname:
                    path = ''.join([root, os.sep, dirname])
                    dirList.append(path)
    print(dirList)
    if query_yes_no("Are you sure you want to delete all these directories AND THEIR CONTENTS under %s?" %rootDir , "yes"):
        nbRemoved = 0
        for dir in dirList:
            print('Deleting dir: %s' % dir)
            shutil.rmtree(dir)
            nbRemoved +=1
        print(nbRemoved, " directories have been deleted")
    return dirList

# root = "/home/matthijs/test/processed"
# name = ["mouths","faces"]

# root = "/home/matthijs/test"
# name = ["processed"]

# deleteDirs(root,name)





def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            print("\t Run 'python fileDirOps.py functionName rootDir (dirNames)'")
            print("\t possible functions: fixnames, frameToTiming, deleteDirs")
            print("\t the third argument is only valid for deleteDirs")
            sys.exit(0)
    # process arguments
    function = args[0]
    rootDir = args[1]
    
    if function == "fixNames":
        fixNames(rootDir)
    elif function == "frameToTiming":
        frameToTiming(rootDir)
    elif function == "deleteDirs":
        if length(args) > 2:
            names = args[2]
        else: print("not enough arguments!")
        deleteDirs(rootDir,names)
    elif function == "convertToGray":
        if length(args) > 2:
            names = args[2]
        else: print("not enough arguments!")
        convertToGrayScale(rootDir,names)
    else:
        print("not a valid function!")
        return -1
    return 0

if __name__ == "__main__":

    # Testing
    import time
    startTime = time.clock() # /home/matthijs/TCDTIMIT/processed
    deleteDirs("/home/matthijs/TCDTIMIT/test/processed/03F/sa1", ["faces_gray","mouths_gray"])
    convertToGrayScale("/home/matthijs/TCDTIMIT/test/processed/03F/sa1", ["faces","mouths"])
    duration = time.clock() - startTime
    print("This took ", duration, " seconds")