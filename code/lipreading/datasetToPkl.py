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
viseme = False
root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/'))
if viseme:
    database_binaryDir = root_dir + 'database_binaryViseme'  # here the raw data was saved.
else:
    database_binaryDir = root_dir + 'database_binary'
    
dataset = "TCDTIMIT"
datasetType = "volunteers";



# set log file
logFile = database_binaryDir + os.sep + dataset + datasetType + "_preprocessing.log"
fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_combinedPKL.addHandler(fh)


## First get the file names of the different sets

testVolunteerNumbers = [13, 15, 21, 23, 24, 25, 28, 29, 30, 31, 34, 36, 37, 43, 47, 51, 54];
testVolunteers = ["Volunteer" + str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers];
lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
allSpeakers = [f for f in os.listdir(database_binaryDir) if os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"]
trainVolunteers = [f if not (f in testVolunteers or f in lipspeakers) else None for f in allSpeakers];
trainVolunteers = [vol for vol in trainVolunteers if vol is not None]

if datasetType == "combined":
    trainingSpeakerFiles = trainVolunteers + lipspeakers
    testSpeakerFiles = testVolunteers
elif datasetType == "volunteers":
    trainingSpeakerFiles = trainVolunteers
    testSpeakerFiles = testVolunteers
else:
    raise Exception("invalid dataset entered")

# add the directory to create paths
trainingSpeakerFiles = sorted([database_binaryDir + os.sep + file for file in trainingSpeakerFiles])
testSpeakerFiles = sorted([database_binaryDir + os.sep + file for file in testSpeakerFiles])



# Now actually preprocess and split
logger_combinedPKL.info("Generating Training data... ")
# generate the data files first
for speakerFile in trainingSpeakerFiles:
    logger_combinedPKL.info("%s", os.path.basename(speakerFile))
    preprocessLipreading.prepLip_one(speakerFile=speakerFile, trainFraction=0.8, validFraction=0.2,
                                     storeDir=os.path.dirname(speakerFile), loadData=False)
logger_combinedPKL.info("Generating Test data... ")
for speakerFile in testSpeakerFiles:
    logger_combinedPKL.info("%s", os.path.basename(speakerFile))
    preprocessLipreading.prepLip_one(speakerFile=speakerFile, trainFraction=0.0, validFraction=0.0,
                                     storeDir=os.path.dirname(speakerFile), loadData=False)