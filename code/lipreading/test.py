import os
import logging, formatting

logger_combined = logging.getLogger('lipreading')
logger_combined.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

viseme = False
dataset = "TCDTIMIT"
root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset))
results_dir = root_dir + "results";
if not os.path.exists(results_dir): os.makedirs(results_dir)
if viseme:
    database_binaryDir = root_dir + '/binaryViseme'
else:
    database_binaryDir = root_dir + '/binary'
datasetType = "volunteers";

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
datasetFiles = [sorted(trainingSpeakerFiles), sorted(testSpeakerFiles)]

import pdb;

pdb.set_trace()

## TEST split train/val/test
from preprocessLipreading import *

X_train, y_train, X_val, y_val, X_test, y_test = prepLip_one(trainingSpeakerFiles[0],
                                                       storeDir=database_binaryDir, trainFraction=0.7,
                                                       validFraction=0.1, storeProcessed=True, verbose=True)

import pdb;

pdb.set_trace()