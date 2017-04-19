from __future__ import print_function
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

import numpy as np
from general_tools import query_yes_no

# 1. remove all specified directories and their contents
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

# stuff for getting relative paths between two directories
def pathsplit(p, rest=[]):
    (h,t) = os.path.split(p)
    if len(h) < 1: return [t]+rest
    if len(t) < 1: return [h]+rest
    return pathsplit(h,[t]+rest)

def commonpath(l1, l2, common=[]):
    if len(l1) < 1: return (common, l1, l2)
    if len(l2) < 1: return (common, l1, l2)
    if l1[0] != l2[0]: return (common, l1, l2)
    return commonpath(l1[1:], l2[1:], common+[l1[0]])

# p1 = main path, p2= the one you want to get the relative path of
def relpath(p1, p2):
    (common,l1,l2) = commonpath(pathsplit(p1), pathsplit(p2))
    p = []
    if len(l1) > 0:
        p = [ '../' * len(l1) ]
    p = p + l2
    return os.path.join( *p )

# 2. copy a dir structure under a new root dir
# copy all mouth files to a new dir, per speaker. Also remove the 'mouths_gray_120' directory, so the files are directly under the videoName folder
# -> processed/lipspeakers

# helpfunction: fix shutil.copytree to allow writing to existing files and directories (http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth#12514470)
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def copyDBFiles(rootDir, names, targetRoot):
    from shutil import copyfile
    dirList = []
    fileList = []
    for root, dirs, files in os.walk(rootDir):
        for dir in dirs:
            for name in names:
                if name in dir:
                    path = ''.join([root, os.sep, dir])
                    dirList.append(path)
        for file in files:
            name, extension = os.path.splitext(file)
            # copy phoneme files as well
            if extension == ".txt":         #TODO Change to .VPHN after renaming done
                path = ''.join([root, os.sep, file])
                fileList.append(path)
    
    print("First 10 files to be copied: ", fileList[0:10])
    print("first 10 dirs to be copied: ", dirList[0:10])

    if query_yes_no("Are you sure you want to copy all these directories from %s to %s?" %(rootDir, targetRoot) , "yes"):
        nbCopiedDirs = 0
        nbCopiedFiles = 0
        
        for dir in dirList:
            relativePath = relpath(rootDir, dir)
            relativePath = relativePath.replace('/mouths_gray_120','')
            dest = ''.join([targetRoot+os.sep+relativePath])
            #print("copying dir:", dir, " to: ", dest)
            copytree(dir, dest)
            nbCopiedDirs +=1
        
        for file in fileList:
            relativePath = relpath(rootDir, file)
            #print("copying file:", file, " to: ", targetRoot+os.sep+relativePath)
            dest = ''.join([targetRoot+os.sep+relativePath])
            copyfile(file, dest)
            nbCopiedFiles +=1
            
        print(nbCopiedDirs, " directories have been copied to ", targetRoot)
        print(nbCopiedFiles, " files have been copied to ", targetRoot)
    return dirList


# need this to traverse directories, find depth
def directories (root):
    dirList = []
    for path, folders, files in os.walk(root):
        for name in folders:
            dirList.append(os.path.join(path, name))
    return dirList

def depth(path):
    return path.count(os.sep)


# extract phonemes for each image, put them in the image name.
# moveToSpeakerDir: - flattens dir structure: to also copy all the jpg and phn files to the speaker dir (so not 1 dir per video)
#                   - also renames: remove speakername, replace '_PHN.txt' by '.vphn'
def addPhonemesToImageNames(videoDir, moveToSpeakerDir = False):
    #print("processing: ", videoDir)
    # videoDir will be the lowest-level directory
    videoName = os.path.basename(videoDir)
    parentName = os.path.basename(os.path.dirname(videoDir))
    validFrames = {}
    phoneme_extension = "_PHN.txt"
    phonemeFile = ''.join([videoDir + os.sep + parentName + "_" + videoName + phoneme_extension])
    # print(phonemeFile)
    with open(phonemeFile) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                frame = str(parts[0])
                phoneme = parts[1] # tuple, (frame, phoneme)
                if frame not in validFrames.keys():
                    validFrames[frame] = [phoneme]       # create list of phn. belonging to this frame
                else:validFrames[frame].append(phoneme)  # add phoneme to the list of phonemes belonging to this frame

    frameImageDict = {}
    # loop through jpgs, get directory with key = frame number, value = imagePath
    for root, dirs, files in os.walk(videoDir):
        for file in files:
            fileName, ext = os.path.splitext(file)
            if ext == ".jpg" and "mouth_gray" in fileName:
                filePath = ''.join([root, os.sep, file])
                videoName = file.split("_")[0]
                frame = file.split("_")[1]  # number-> after first underscore
                frameImageDict[str(frame)] = filePath

    for frame in validFrames.keys():
        for phoneme in validFrames[frame]:
            # for each phoneme, create a copy of the image belonging to this frame
            try:imagePath = frameImageDict[frame]
            except: return videoDir #frame not found, means something went wrong in the extraction -> delete dir and extract again
            videoName = os.path.basename(imagePath).split("_")[0]
            newFilePath = ''.join([os.path.dirname(imagePath), os.sep, videoName, "_", frame, "_", phoneme, ".jpg"])
            shutil.copy2(imagePath, newFilePath)
        #print(videoDir)

    # delete the source images
    for imagePath in frameImageDict.values():
        os.remove(imagePath)

    return None  #successful -> nothing to dbe deleted


# now traverse the database tree and rename  files in all the directories
def addPhonemesToImagesDB(rootDir, moveToSpeakerDir = False):
    badDirs = []

    dirList = []
    for dir in directories(rootDir):
        # print(dir)
        # print(relpath(rootDir,dir))
        # print(depth(relpath(rootDir,dir)))
        if depth(relpath(rootDir, dir)) == 2:
            dirList.append(dir)
    print("First 10 directories to be processed: ", dirList[0:10])
    for dir in dirList:
        result = addPhonemesToImageNames(dir, moveToSpeakerDir=moveToSpeakerDir)
        if result != None: badDirs.append(result)
        if moveToSpeakerDir: shutil.rmtree(dir)
    return badDirs


# helpfunction
from phoneme_set import phoneme_set_39
def getPhonemeNumberMap():
    return phoneme_set_39


from general_tools import *
        
if __name__ == "__main__":

    # use this to copy the grayscale files from 'processDatabase' to another location, and fix their names with phonemes
    # then convert to files useable by lipreading network
    
    processedDir = os.path.expanduser("~/TCDTIMIT/lipreading/processed")
    databaseDir = os.path.expanduser("~/TCDTIMIT/combinedSR/TCDTIMIT/database2")
    
    # 1. copy mouths_gray_120 images and PHN.txt files to targetRoot. Move files up from their mouths_gray_120 dir to the video dir (eg sa1)
    print("Copying mouth_gray_120 directories to database location...")
    copyDBFiles(processedDir, ["mouths_gray_120"], databaseDir)
    print("-----------------------------------------")
    
    # # 2. extract phonemes for each image, put them in the image name
    # if two phonemes for one frame, copy the image so we have 2 times the same frame, but with a different phoneme (in the name)
    # # has to be called against the 'database' directory
    print("Adding phoneme to filenames...")
    badDirs = addPhonemesToImagesDB(databaseDir, moveToSpeakerDir=False)
    print("-----------------------------------------")

    saveToPkl('./badDirs2.pkl', badDirs)
    print(len(badDirs))
    import pdb;pdb.set_trace()


