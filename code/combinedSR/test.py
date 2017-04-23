import os

dataset = "TCDTIMIT"
root_dir = os.path.expanduser("~/TCDTIMIT/combinedSR/")
store_dir = root_dir + dataset + "/results"
if not os.path.exists(store_dir): os.makedirs(store_dir)

if not os.path.exists(store_dir): os.makedirs(store_dir)
database_binaryDir = root_dir + dataset + '/binary'
datasetType = "combined";


# just get the names
testVolunteerNumbers = ["13F", "15F", "21M", "23M", "24M", "25M", "28M", "29M", "30F", "31F", "34M", "36F", "37F", "43F", "47M", "51F", "54M"];
testVolunteers = [str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers];
lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
allSpeakers = [f for f in os.listdir(database_binaryDir) if
               os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"]
trainVolunteers = [f for f in allSpeakers if not (f in testVolunteers or f in lipspeakers)];

if datasetType == "combined":
    trainingSpeakerFiles = trainVolunteers + lipspeakers
    testSpeakerFiles = testVolunteers
elif datasetType == "volunteers":
    trainingSpeakerFiles = trainVolunteers
    testSpeakerFiles = testVolunteers
else:
    raise Exception("invalid dataset entered")
datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]

import pdb;pdb.set_trace()