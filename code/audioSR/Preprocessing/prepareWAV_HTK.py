import os
import string
import sys
import exceptions
import errno

# Search a dir for wav files, generate files with paths so HTK can process them

# search recursively through a directory, collecting paths of all wav files.
# Then a .scp file (just a txt really) is written, containing the wav path, a space, and the mfc path.
# This mfc path is the destination that the HTK toolkit will write the output MFC file to.
# so it will look like this:
#  wavPath_file1/ mfcPath_file1/
#  wavPath_file2/ mfcPath_file2/
#  etc...
# After executing this file, run 'HCopy -A -D -T 1 -C wav_config -S wavAndMFCCPaths.scp' in the directory where the .scp file is stored
# after the mfc files are generated, I manually copied them back to the data folder, so they are stored together with the wav and label files

# If that gives an error (eg "Input file is not in RIFF format"), you can run fixWavs.py, using the output files from this script as input to fixWavs.

#  see http://www.voxforge.org/home/dev/acousticmodels/linux/create/htkjulius/tutorial/data-prep/step-5

def getWavMFCCLocations(baseDir):
    dirs = []
    wavs = []
    wavAndMFCC = []
    for root, directories, filenames in os.walk(baseDir):
        for directory in directories:
            # delete empty directories
            dirPath = os.path.join(root,directory)
            try:
                os.rmdir(dirPath)
            except OSError as ex:
                if ex.errno == errno.ENOTEMPTY:
                    dirs.append(dirPath)
        for file in filenames:
            path = os.path.join(root, file)
            # delete  empty files (or almost empty, eg 44 bytes)
            if os.stat(path).st_size <= 44:
                os.remove(path)
            if os.path.splitext(path)[1].lower() == '.wav':
                wavs.append(path)

                # change store dir of MFCC file of this WAV
                thisDir = os.path.dirname(path)
                relPath = os.path.relpath(thisDir, baseDir)
                newPath = baseDir + os.sep + "mfc" + os.sep + relPath
                # Create directory structure if needed
                if not os.path.exists(newPath):
                    os.makedirs(newPath)
                pathMFCC = newPath + os.sep + os.path.splitext(os.path.basename(path))[0] + ".mfc" #change extension

                wavAndMFCC.append(path + " " + pathMFCC)

    wavs.sort(key=string.lower)
    dirs.sort(key=string.lower)
    wavAndMFCC.sort(key=string.lower)
    return dirs, wavs, wavAndMFCC



def prepareWAV_HTK(baseDir = os.path.expanduser('~/TCDTIMIT/TIMIT/TIMIT'), fileStoreDir = os.path.expanduser('~/TCDTIMIT/TIMIT/TIMIT') ):
    from helpFunctions import *

    print "Searching for WAVs in: ", baseDir

    dirs, wavs, wavAndMFCC = getWavMFCCLocations(baseDir)
    print "Dirs: ", dirs[0:3]
    print "Wavs: ", wavs[0:3]
    print "Wav + MFCC: ", wavAndMFCC[0:3]
    print "Number of files: ", len(wavs)

    # Write the files
    print "Writing the files..."

    wavFilePath = fileStoreDir + os.sep + 'wavPaths.txt'
    writeToTxt(wavs, wavFilePath)

    wavAndMFCCFilePath = fileStoreDir + os.sep + 'wavAndMFCCPaths.scp'
    writeToTxt(wavAndMFCC, wavAndMFCCFilePath)

    # write wav_config file
    # from the HTK manual, p31: In brief, they specify that the target parameters are to be MFCC using C 0 as the energy
    # component, the frame period is 10msec (HTK uses units of 100ns), the output should be saved in
    # compressed format, and a crc checksum should be added. The FFT should use a Hamming window
    # and the signal should have first order preemphasis applied using a coefficient of 0.97. The filterbank
    # should have 26 channels and 12 MFCC coefficients should be output. The variable ENORMALISE is
    # by default true and performs energy normalisation on recorded audio files. It cannot be used with
    # live audio and since the target system is for live audio, this variable should be set to false.
    wav_config = [
        "SOURCEFORMAT = WAV",
        "TARGETKIND = MFCC_0_D",
        "TARGETRATE = 100000.0",
        "SAVECOMPRESSED = T",
        "SAVEWITHCRC = T",
        "WINDOWSIZE = 250000.0",
        "USEHAMMING = T",
        "PREEMCOEF = 0.97",
        "NUMCHANS = 26",
        "CEPLIFTER = 22",
        "NUMCEPS = 12"
    ]
    wavConfigPath = fileStoreDir + os.sep + 'wav_config'
    writeToTxt(wav_config, wavConfigPath)

    print "Done."
    print "List of wavs has been written to:        ", wavFilePath
    print "List of wavs + MFCC has been written to: ", wavAndMFCCFilePath
    print "Wav_config for HTK has been written to: ", wavConfigPath
    print "Now run 'HCopy -A -D -T 1 -C wav_config -S wavAndMFCCPaths.scp' in the directory where the .scp file is stored"
    return 0


if __name__ == '__main__':
    # SPECIFY Default SOURCE FOLDER
    # TIMIT:
    baseDir = os.path.expanduser('~/TCDTIMIT/TIMIT/TIMIT')
    # TCDTIMIT:
    # baseDir = os.path.expanduser('/media/matthijs/TOSHIBA_EXT/TCDTIMIT')
    # TIMIT 2, maybe incomplete:
    # TIMIT: #'~/TCDTIMIT/TIMITaudio/wav')  #

    # MFC files will be stored in 'mfc' folder on the same level as the 'wav' folder
    fileStoreDir = baseDir  # os.path.dirname(baseDir) # store one level above.

    nbArgs = len(sys.argv)
    if (nbArgs ==1):
        prepareWAV_HTK(sys.argv[1])
    elif (nbArgs==2):
        prepareWAV_HTK(sys.argv[1], sys.argv[2])
    else:
        print "ERROR, too many arguments"
