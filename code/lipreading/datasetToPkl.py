import os
import logging, formatting
import preprocessLipreading

logger_combinedPKL = logging.getLogger('PrepTCDTIMIT')
logger_combinedPKL.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_combinedPKL.addHandler(ch)



### The raw data was already saved in pkl files by FileDirOps, now we need to preprocess it (normalize and split in train/val/test sets)
dataset = "TCDTIMIT"
datasetType = "lipspeakers";
viseme = False
root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset))
database_binaryDir = root_dir + "/binary" # the raw data
if not os.path.exists(database_binaryDir): os.makedirs(database_binaryDir)

# set log file
logFile = database_binaryDir + os.sep + datasetType + "_preprocessing.log"
fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_combinedPKL.addHandler(fh)


## First get the file names of the different sets

# just get the names
testVolunteerNumbers = ["13F", "15F", "21M", "23M", "24M", "25M", "28M", "29M", "30F", "31F", "34M", "36F", "37F",
                        "43F", "47M", "51F", "54M"];
testVolunteers = sorted([str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers])
lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
allSpeakers = sorted([f for f in os.listdir(database_binaryDir) if
                      os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"])
trainVolunteers = sorted([f for f in allSpeakers if not (f in testVolunteers or f in lipspeakers)])

if datasetType == "combined":
    trainingSpeakerFiles = trainVolunteers + lipspeakers
    testSpeakerFiles = testVolunteers
elif datasetType == "volunteers":
    trainingSpeakerFiles = trainVolunteers
    testSpeakerFiles = testVolunteers
else:
    raise Exception("invalid dataset entered")


# add the directory to create paths
trainingSpeakerFiles = sorted([file for file in trainingSpeakerFiles])
testSpeakerFiles = sorted([file for file in testSpeakerFiles])


# Now actually preprocess and split
logger_combinedPKL.info("Generating Training data... ")
# generate the data files first
for speakerFile in trainingSpeakerFiles:
    logger_combinedPKL.info("%s", os.path.basename(speakerFile))
    preprocessLipreading.prepLip_one(speakerFile=speakerFile, trainFraction=0.8, validFraction=0.2,
                                     sourceDataDir=database_binaryDir, loadData=False, viseme=viseme, storeProcessed=True)
logger_combinedPKL.info("Generating Test data... ")
for speakerFile in testSpeakerFiles:
    logger_combinedPKL.info("%s", os.path.basename(speakerFile))
    preprocessLipreading.prepLip_one(speakerFile=speakerFile, trainFraction=0.0, validFraction=0.0,
                                     sourceDataDir=database_binaryDir, loadData=False, viseme=viseme, storeProcessed=True)