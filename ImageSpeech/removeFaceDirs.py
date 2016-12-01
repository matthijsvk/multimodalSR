import zipfile, os.path
import concurrent.futures
import threading

import os, errno
import subprocess
import shutil

from helpFunctions import *

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
    if query_yes_no("Are you sure you want to delete all these directories AND THEIR CONTENTS under %s?" %rootDir , "yes"):
        for dir in dirList:
            print('Deleting dir: %s' % dir)
            shutil.rmtree(dir)
    return dirList

#root = "/home/matthijs/TCDTIMIT/processed2/volunteers"

# root = "/home/matthijs/test/processed"
# name = ["mouths","faces"]

root = "/home/matthijs/test"
name = ["processed"]
deleteDirs(root,name)
