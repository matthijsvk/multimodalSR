import os, errno
import subprocess
import sys

# We need to execute this command for every wav file we find:
# mplayer \
#   -quiet \
#   -vo null \
#   -vc dummy \
#   -ao pcm:waveheader:file="audio_FIXED.wav" audio_BROKEN.wav
# (see http://en.linuxreviews.org/HOWTO_Convert_audio_files)

# Read a file that contains the paths of all our .wav files
# Then, for each wav file, get the path where the fixed version should be stored
# Then generate the fixed files.
from audioPhonemeRecognition.processDataset.fixDataset.old.prepareWAV_HTK import prepareWAV_HTK


def readFile(filename):
    with open(filename, "r") as ins:
        array = []
        for line in ins:
            line = line.strip('\n')  # strip newlines
            if len(line) > 1:  # don't save the dots lines
                    array.append(line)

    return array

def fixWav(wavPath, fixedWavPath):

    name = os.path.basename(wavPath)
    if not os.path.exists(os.path.dirname(fixedWavPath)):  # skip already existing videos (it is assumed the exist if the directory exists)
        os.makedirs(os.path.dirname(fixedWavPath))

    if not os.path.exists(fixedWavPath):
        command = ['mplayer',
                   '-quiet',
                   '-vo', 'null',
                   '-vc', 'dummy',
                   '-ao', 'pcm:waveheader:file='+fixedWavPath,
                   wavPath]

        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
        return 1
    else:
        return 0

def getFixedWavPath(path, baseDir, fixedDir):
    thisDir = os.path.dirname(path)
    relPath = os.path.relpath(thisDir, baseDir)
    newPath = ''.join([fixedDir,os.sep,relPath,os.sep,os.path.basename(path)])
    return newPath


def fixWavs(baseDir, fixedDir):
    print("Fixing WAV files in ", baseDir, " and storing to: ", fixedDir)

    # generate file that contains paths to wav files
    print("Searching for WAVs in: ", baseDir)
    prepareWAV_HTK(baseDir, baseDir)

    pathsFile = baseDir + os.sep + 'wavPaths.txt'
    wavPaths = readFile(pathsFile)

    # fix files, store them under fixedDir
    nbFixed=0
    for wavPath in wavPaths:
        fixedWavPath = getFixedWavPath(wavPath, baseDir, fixedDir)
        fixWav(wavPath, fixedWavPath)

        nbFixed+=1
        if (nbFixed % 100 == 0):
            print("Fixed ", nbFixed, "out of",len(wavPaths))

    return 0

