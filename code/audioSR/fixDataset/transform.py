import argparse
import csv
import os
import subprocess
import sys

from tqdm import tqdm

from helpFunctions import resample
from helpFunctions.writeToTxt import writeToTxt
from phoneme_set import phoneme_set_39, phoneme_set_61_39

debug = False

# input dir should be the dir just above 'TRAIN' and 'TEST'. It expects
# example usage: python transform.py phonemes -i /home/matthijs/TCDTIMIT/TIMIT/original/TIMIT/ -o /home/matthijs/TCDTIMIT/TIMIT/processed


##### Load Data  ######
# find all files of a type under a directory, recursive
def load_wavPhn(rootDir):
    wavs = loadWavs(rootDir)
    phns = loadPhns(rootDir)
    return wavs, phns


def loadWavs(rootDir):
    wav_files = []
    for dirpath, dirs, files in os.walk(rootDir):
        for f in files:
            if (f.lower().endswith(".wav")):
                wav_files.append(os.path.join(dirpath, f))
    return sorted(wav_files)


def loadPhns(rootDir):
    phn_files = []
    for dirpath, dirs, files in os.walk(rootDir):
        for f in files:
            if (f.lower().endswith(".phn")):
                phn_files.append(os.path.join(dirpath, f))
    return sorted(phn_files)


# generates for example: dstDir/TIMIT/TRAIN/DR2/MTAT1/SX59.PHN'
# from srcPath = someDir/TIMIT/TRAIN/DR2/MTAT1/SX239.PHN')
def getDestPath(srcPath, dstDir):
    filename = os.path.basename(srcPath)

    speakerPath = os.path.dirname(srcPath)
    speaker = os.path.basename(speakerPath)

    regionPath = os.path.dirname(speakerPath)
    region = os.path.basename(regionPath)

    setPath = os.path.dirname(regionPath)
    set = os.path.basename(setPath)

    timitPath = os.path.dirname(setPath)
    timit = os.path.basename(timitPath)

    dstPath = os.path.join(dstDir, timit, set, region, speaker, filename)
    return dstPath


###  TRANSFORM FUNCTIONS ###
# create a wav file with proper headers
def transformWav(wav_file, dstPath):
    output_dir = os.path.dirname(dstPath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(dstPath):
        command = ['mplayer',
                   '-quiet',
                   '-vo', 'null',
                   '-vc', 'dummy',
                   '-ao', 'pcm:waveheader:file=' + dstPath,
                   wav_file]

        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE

        # TODO this line is commented out to enable parallel file processing; uncomment if you need to access the file directly after creation
        # subprocess.Popen.wait(p) # wait for completion
        return 1
    else:
        return 0


# generate new .phn file with mapped phonemes (from 61, to 39 -> see dictionary in phoneme_set.py)
def transformPhn(phn_file, dstPath):
    output_dir = os.path.dirname(dstPath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(dstPath):
        # extract label from phn
        phn_labels = []
        with open(phn_file, 'rb') as csvfile:
            phn_reader = csv.reader(csvfile, delimiter=' ')
            for row in phn_reader:
                start, stop, label = row[0], row[1], row[2]

                if label not in phoneme_set_39.keys():  # map from 61 to 39 phonems using dict
                    label = phoneme_set_61_39.get(label)

                classNumber = label  # phoneme_set_39[label] - 1 # get class number
                phn_labels.append([start, stop, classNumber])

        # print phn_labels
        # print phn_labels
        writeToTxt(phn_labels, dstPath)


###########  High Level Functions ##########
# just loop over all the found files
def transformWavs(args):
    srcDir = args.srcDir
    dstDir = args.dstDir

    print("src: ", srcDir)
    print("dst: ", dstDir)
    srcWavs = loadWavs(srcDir)
    srcWavs.sort()

    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    resampled = []
    # transform: fix headers and resample. Use 2 seperate loops to prevent having to wait for the fixed file to be written
    #               therefore also don't wait until completion in transformWav (the Popen.wait(p) line)
    print("FIXING WAV HEADERS AND COPYING TO ", dstDir, "...")
    for srcPath in tqdm(srcWavs, total=len(srcWavs)):
        dstPath = getDestPath(srcPath, dstDir)
        resampled.append(dstPath)
        transformWav(srcPath, dstPath)
        if debug: print(srcPath, dstPath)

    print("RESAMPLING TO 16kHz...")
    for dstPath in tqdm(resampled, total=len(resampled)):
        resample.resampleWAV(dstPath, dstPath, out_fr=16000.0, q=1.0)  # resample to 16 kHz from 48kHz

        ## TODO USING resampy library: about 4x faster, but sometimes weird crashes...
        # in_fr, in_data = wavfile.read(dstPath)
        # in_type = in_data.dtype
        # in_data = in_data.astype(float)
        # # x is now a 1-d numpy array, with `sr_orig` audio samples per second
        # # We can resample this to any sampling rate we like, say 16000 Hz
        # y_low = resampy.resample(in_data, in_fr, 16000)
        # y_low = y_low.astype(in_type)
        # wavfile.write(dstPath, 16000, y_low)


def transformPhns(args):
    srcDir = args.srcDir
    dstDir = args.dstDir
    srcPhns = loadPhns(srcDir)

    print("Source Directory: ", srcDir)
    print("Destination Directory: ", dstDir)

    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    for srcPath in tqdm(srcPhns, total=len(srcPhns)):
        dstPath = getDestPath(srcPath, dstDir)
        # print("reading from: ", srcPath)
        # print("writing to: ", dstPath)
        transformPhn(srcPath, dstPath)


## help functions ###
def readPhonemeDict(filePath):
    d = {}
    with open(filePath) as f:
        for line in f:
            (key, val) = line.split()
            d[int(key)] = val
    return d


def checkDirs(args):
    if 'dstDir' in args and not os.path.exists(args.dstDir):
        os.makedirs(args.dstDir)
    if 'srcDir' in args and not os.path.exists(args.srcDir):
        raise Exception('Can not find source data path')


#################  PARSER #####################
def prepare_parser():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    ## TRANSFORM ##
    phn_parser = sub_parsers.add_parser('phonemes')
    phn_parser.set_defaults(func=transformPhns)
    phn_parser.add_argument('-i', '--srcDir',
                            help="the directory storing source data",
                            required=True)
    phn_parser.add_argument('-o', '--dstDir',
                            help="the directory store output data",
                            required=True)
    ## TRANSFORM ##
    wav_parser = sub_parsers.add_parser('wavs')
    wav_parser.set_defaults(func=transformWavs)
    wav_parser.add_argument('-i', '--srcDir',
                            help="the directory storing source data",
                            required=True)
    wav_parser.add_argument('-o', '--dstDir',
                            help="the directory store output data",
                            required=True)
    return parser


if __name__ == '__main__':
    arg_parser = prepare_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    checkDirs(args)
    args.func(args)
