from __future__ import print_function

import warnings
from time import gmtime, strftime
from pprint import pprint  #printing properties of networkToRun objects
#pprint(vars(a))

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting
import os
from tqdm import tqdm

logger_combined = logging.getLogger('combined')
logger_combined.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_combined.addHandler(ch)

# File logger: see below META VARIABLES

import time
program_start_time = time.time()

print("\n * Importing libraries...")
from combinedNN_tools import *
from general_tools import *
from trackMemory import *
import preprocessingCombined
import traceback

###################### Script settings #######################################
resultsPath = os.path.expanduser('~/TCDTIMIT/combinedSR/TCDTIMIT/results/allEvalResults.pkl')

logToFile = True; overwriteResults = False

# if you wish to force retrain of networks, set justTest to False, forceTrain in main() to True, and overwriteSubnets to True.
# if True, and justTest=False, even if a network exists it will continue training. If False, it will just be evaluated
forceTrain = False
autoTrain = False # if true, automatically starts training of a network of which we couldn't find results (probably because it hadn't been trained yet)

# use when you want to train a combined network or lipreading-LSTM when you've created a better subnetwork.
# You can load the new subnet (eg lipreading CNN or audio network) in the combined network, and retrain it. You dont' have to redo everything.
overwriteSubnets = False

# JustTest: If True, mainGetResults just runs over the trained networks. If a network doesn't exist, it's skipped
#           If False, ask user to train networks, then start training networks that don't exist.
justTest=True

# TODO: store these in the NetworkParams class instead of ugly global vars...
# (only for lipreading networks): use viseme labels and classification ipv phoneme
viseme=False
getConfusionMatrix = True  # if True, stores confusionMatrix where the .npz and train_info.pkl are stored
# use this for testing with reduced precision. It converts the network weights to float16, then back to float32 for execution.
# This amounts to rounding. Performance should hardly be impacted.
ROUND_PARAMS = False

# use this to test trained networks on the test dataset with noise added.
# This data is generated using audioSR/fixDataset/mergeAudiofiles.py + audioToPkl_perVideo.py and combinedSR/dataToPkl_lipspeakers.py
# It is loaded in in combinedNN_tools/finalEvaluation (also just before training in 'train'. You could also generate noisy training data and train on that, but I haven't tried that
withNoise = False
noiseTypes=['white','voices']
ratio_dBs = [0,-3,-5,-10]

###################### Script code #######################################

def main():
    networkList = [
        # # # # # # ### LIPREADING ###
        # # # # # # # # # CNN
        # networkToRun(runType="lipreading", CNN_NETWORK="google",LIP_RNN_HIDDEN_LIST=None, forceTrain=forceTrain),
        # networkToRun(runType="lipreading", CNN_NETWORK="resnet50",LIP_RNN_HIDDEN_LIST=None, forceTrain=forceTrain),
        # networkToRun(runType="lipreading", CNN_NETWORK="cifar10", LIP_RNN_HIDDEN_LIST=None, forceTrain=forceTrain),
        #
        # # # # # # # # # #
        # # # # # # # # # # # CNN-LSTM -> by default only the LSTM part is trained (line 713 in build_functions in combinedNN_tools.py
        # # # # # # # # # # #          -> you can train everything (also CNN parameters), but only do this after the CNN-LSTM has trained with fixed CNN parameters, otherwise you'll move far out of your optimal point
        # # # # # compare number of LSTM units
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[8],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain,LR_start = 0.01),
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[64],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain,LR_start = 0.01),
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[512,512],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain, LR_start=0.01),
        #
        # # # # # compare with conv
        # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[8], overwriteSubnets=overwriteSubnets,
        #                forceTrain=forceTrain,LR_start = 0.01),
        # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[64], overwriteSubnets=overwriteSubnets,
        #                forceTrain=forceTrain,LR_start = 0.01),
        # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256], overwriteSubnets=overwriteSubnets,
        #                forceTrain=forceTrain),
        # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], overwriteSubnets=overwriteSubnets,
        #                forceTrain=forceTrain),
        # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[512, 512], overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        #
        # # resnet 50 + LSTM
        # networkToRun(runType="lipreading", CNN_NETWORK="resnet50", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
        #              overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        #
        #
        # # # # # ### AUDIO ###  -> see audioSR/RNN.py, there it can run in batch mode which is much faster
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),#,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              audio_dataset="combined", test_dataset="TCDTIMIT"),#, forceTrain=True)
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),  # , forceTrain=True)
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),#,forceTrain=True),

        # run TCDTIMIT-trained network on lipspeakers, and on the real TCDTIMIT test set (volunteers)
        networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                     audio_dataset="TCDTIMIT", test_dataset="TIMIT"),  # , forceTrain=True)  % 66.75 / 89.19
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMITvolunteers"),  # ,forceTrain=True),


        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),

        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),


        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256], dataset="TIMIT", audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32, 32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024,1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8,8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32,32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64,64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256,256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512,512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024, 1024, 1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8,8,8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32,32,32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64,64,64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256,256,256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512,512,512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024,1024,1024,1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        # #get the MFCCS
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=13),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=26),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT", nbMFCCs=39),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=120),

        # # # ### COMBINED ###
        # # # # lipspeakers
        # # different number of FC softmax dense for DENSE
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
        #                DENSE_HIDDEN_LIST=[256], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
        #                DENSE_HIDDEN_LIST=[256,256],overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
        #                 DENSE_HIDDEN_LIST=[256,256,256],overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
        #                DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # # different number of FC softmax dense for CONV
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=None,
        #              DENSE_HIDDEN_LIST=[256], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=None,
        #              DENSE_HIDDEN_LIST=[256, 256], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=None,
        #              DENSE_HIDDEN_LIST=[256, 256, 256], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=None,
        #              DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # impact of small audio network
        # networkToRun(AUDIO_LSTM_HIDDEN_LIST=[32,32],
        #                cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],
        #                DENSE_HIDDEN_LIST=[512, 512, 512],overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # # lipRNN features as output
        # networkToRun(AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],
        #              DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # # lipreading: phoneme as outputs not lipRNNfeatures.
        # # # Also see if DENSE_HIDDEN_LIST has influence (Should as we give more easily interpretable results as output (phonen probs instead of RNN features)
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        # #              DENSE_HIDDEN_LIST=[512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # # with dense cnn features
        # # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        # #              DENSE_HIDDEN_LIST=[512], overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # audio outputs are phonemes
        # networkToRun(audio_features="dense", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
        #              lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512],  overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(audio_features="dense",
        #              cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],  lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], overwriteSubnets=overwriteSubnets,  forceTrain=forceTrain),
        # networkToRun(audio_features="dense",
        #              cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",
        #              DENSE_HIDDEN_LIST=[512, 512, 512],  overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        #
        # # # # # SUBNET TRAINING
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512],
        #              allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,forceTrain=forceTrain),
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        #
        # # audio output are phonemes
        # networkToRun(audio_features="dense",cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        # networkToRun(audio_features="dense", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        # networkToRun(audio_features="dense",cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True, overwriteSubnets=overwriteSubnets,
        #              forceTrain=forceTrain),
        #
        # # VOLUNTEERS
        # # test lipspeaker trained networks on volunteers
        # networkToRun(runType="lipreading", CNN_NETWORK="google", LIP_RNN_HIDDEN_LIST=None,
        #              test_dataset="TCDTIMITvolunteers", LR_start=0.01),
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], test_dataset="TCDTIMITvolunteers"),
        #
        #
        # # ## lipreading
        # networkToRun(runType="lipreading", LIP_RNN_HIDDEN_LIST=None,
        #              dataset="TCDTIMITvolunteers", overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # networkToRun(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
        #              dataset="TCDTIMITvolunteers", overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        # # networkToRun(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],
        # #              dataset="TCDTIMITvolunteers", overwriteSubnets=overwriteSubnets, forceTrain=forceTrain),
        #
        # # # #audio test on volunteers
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256], dataset="TCDTIMITvolunteers"),
        #
        # # # # Combined
        # #lipreading full dense, audio raw
        # networkToRun(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], dataset="TCDTIMITvolunteers"),
        # # full raw, allowSubnetTraining
        # networkToRun(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",  #audio also raw
        #              DENSE_HIDDEN_LIST=[512, 512, 512], dataset="TCDTIMITvolunteers", allowSubnetTraining=True),

        # # ATTENTION
        # networkToRun(audio_features='raw', AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",
        #              DENSE_HIDDEN_LIST=[512, 512, 512], combinationType='attention',
        #              allowSubnetTraining=False, overwriteSubnets=overwriteSubnets, forceTrain=forceTrain,
        #              addNoisyAudio=True, LR_start=0.01)

        # TODO: there are 3 trained attention networks -> get these manually
        # 1. PC, PC just with all audio.
        # 2. RF, RF first normal audio, then retrained on white noise audio
        # 3. RF, RF first normal audio, then retrained on white noise audio, then retrained on voices
    ]

    # # # ATTENTION
    # With attention networks (only works with pre-classifying lipreading and audio networks
    # The atention network then combines the output of both optimally
    # # audio PF + lip PC (CNN RF)
    # networkToRun(audio_features='dense', AUDIO_LSTM_HIDDEN_LIST=[256, 256],
    #              cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="dense",
    #              DENSE_HIDDEN_LIST=[512, 512, 512], combinationType='attention',
    #              allowSubnetTraining=False, overwriteSubnets=False, forceTrain=forceTrain,
    #              addNoisyAudio=True), #make sure to also set withNoise to True above the main()
    noisyAudioNetworkList = [
        networkToRun(audio_features='raw', AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                     cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256], lipRNN_features="raw",
                     DENSE_HIDDEN_LIST=[512, 512, 512], combinationType='attention',
                     allowSubnetTraining=False, overwriteSubnets=overwriteSubnets, forceTrain=True,#forceTrain,
                     addNoisyAudio=True, LR_start=0.01)  # make sure to also set withNoise to True above the main()
    ]

    #Use this if you want only want to start training if the network doesn't exist
    if withNoise:
        allResults = []
        for noiseType in noiseTypes:
            for ratio_dB in ratio_dBs:
                results, resultsPath = mainGetResults(networkList, withNoise, noiseType, ratio_dB)
                allResults.append(results)
                # print(allResults)
                # import pdb;pdb.set_trace()

        allNoisePath = os.path.expanduser('~/TCDTIMIT/resultsNoisy.pkl')
        exportResultsToExcelManyNoise(allResults, allNoisePath)
    else:
        results, resultsPath = mainGetResults(networkList)
        print("\n got all results")
        exportResultsToExcel(results, resultsPath)

    # # # use this if you want to force run the network on train sets. If justTest==True, it will only evaluate performance on the test set
    runManyNetworks(networkList, withNoise=withNoise)


    # # For training networks with noisy audio:
    #runManyNetworks(noisyAudioNetworkList, withNoise = True, noiseType = noiseTypes, ratio_dB = ratio_dBs)

# this loads the specified results from networks in networkList
def mainGetResults(networkList, withNoise=False, noiseType='white', ratio_dB=0):
    resultsType = ("roundParams" if ROUND_PARAMS else "") + (
    "_Noise" + noiseType + "_" + str(ratio_dB) if withNoise else "")
    if resultsType == "": resultsType = "default"
    resultsPath = os.path.expanduser('~/TCDTIMIT/storedResults') + resultsType + ".pkl"

    try: prevResults = unpickle(resultsPath)
    except: prevResults = {}

    results = getManyNetworkResults(networkList, resultsType=resultsType,
                                    roundParams=ROUND_PARAMS,
                                    withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB)

    prevResults.update(results)
    saveToPkl(resultsPath, prevResults)
    return prevResults,resultsPath

def exportResultsToExcel(results, path):
    storePath = path.replace(".pkl",".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)


    for runType in results.keys()[1:]: #audio, lipreading, combined:
        worksheet = workbook.add_worksheet(runType) # one worksheet per runType, but then everything is spread out...
        row = 0

        allNets = results[runType]

        # get and write the column titles

        # get the number of parameters. #for audio, only 1 value. For combined/lipreadin: lots of values in a dictionary
        try:nb_paramNames = allNets.items()[0][1]['nb_params'].keys() #first key-value pair, get the value ([1]), then get names of nbParams (=the keys)
        except: nb_paramNames =['nb_params']
        startVals = 4 + len(nb_paramNames)  # column number of first value

        colNames = ['Network Full Name', 'Network Name','Dataset','Test Dataset']+nb_paramNames + ['Test Cost', 'Test Accuracy', 'Test Top 3 Accuracy', 'Validation accuracy']
        for i in range(len(colNames)):
            worksheet.write(0, i, colNames[i])

        # write the data for each network
        for netName in allNets.keys():
            row += 1

            thisNet = allNets[netName]
            # write the path and name
            worksheet.write(row, 0, os.path.basename(netName))  # netName)
            worksheet.write(row, 1, thisNet['niceName'])
            if runType == 'audio':
                worksheet.write(row, 2, thisNet['audio_dataset'])
                worksheet.write(row, 3, thisNet['test_dataset'])
            else:
                worksheet.write(row, 2, thisNet['dataset'])
                worksheet.write(row, 3, thisNet['test_dataset'])

            # now write the params
            try: vals = thisNet['nb_params'].values()  # vals is list of [test_cost, test_acc, test_top3_acc]
            except: vals = [thisNet['nb_params']]
            for i in range(len(vals)):
                worksheet.write(row, 4 + i, vals[i])

            # now write the values
            vals = thisNet['values']  # vals is list of [test_cost, test_acc, test_top3_acc]
            for i in range(len(vals)):
                worksheet.write(row, startVals + i, vals[i])

    workbook.close()

    logger_combined.info("Excel file stored in %s", storePath)


def exportResultsToExcelManyNoise(resultsList, path):
    storePath = path.replace(".pkl", ".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)

    storePath = path.replace(".pkl", ".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)

    row = 0

    if len(resultsList[0]['audio'].keys()) > 0: thisRunType = 'audio'
    if len(resultsList[0]['lipreading'].keys()) > 0: thisRunType = 'lipreading'
    if len(resultsList[0]['combined'].keys()) > 0: thisRunType = 'combined'
    worksheetAudio = workbook.add_worksheet('audio'); audioRow = 0
    worksheetLipreading = workbook.add_worksheet('lipreading'); lipreadingRow = 0
    worksheetCombined = workbook.add_worksheet('combined'); combinedRow = 0

    for r in range(len(resultsList)):
        results = resultsList[r]
        noiseType = results['resultsType']

        for runType in results.keys()[1:]:
            if len(results[runType]) == 0: continue
            if runType == 'audio': worksheet = worksheetAudio; row = audioRow
            if runType == 'lipreading': worksheet = worksheetLipreading; row = lipreadingRow
            if runType == 'combined': worksheet = worksheetCombined; row = combinedRow

            allNets = results[runType]

            # write the column titles
            startVals= 5
            colNames = ['Network Full Name', 'Network Name', 'Dataset', 'Test Dataset', 'Noise Type', 'Test Cost', 'Test Accuracy',
                        'Test Top 3 Accuracy']
            for i in range(len(colNames)):
                worksheet.write(0, i, colNames[i])

            # write the data for each network
            for netName in allNets.keys():
                row += 1

                thisNet = allNets[netName]
                # write the path and name
                worksheet.write(row, 0, os.path.basename(netName)) #netName)
                worksheet.write(row, 1, thisNet['niceName'])
                if runType == 'audio':
                    worksheet.write(row, 2, thisNet['audio_dataset'])
                    worksheet.write(row, 3, thisNet['test_dataset'])
                else:
                    worksheet.write(row, 2, thisNet['dataset'])
                    worksheet.write(row, 3, thisNet['test_dataset'])
                worksheet.write(row, 4, noiseType)


                # now write the values
                vals = thisNet['values']  # vals is list of [test_cost, test_acc, test_top3_acc]
                for i in range(len(vals)):
                    worksheet.write(row, startVals + i, vals[i])

            if runType == 'audio': audioRow = row
            if runType == 'lipreading': lipreadingRow = row
            if runType == 'combined': combinedRow = row

        row += 1

    workbook.close()

    logger_combined.info("Excel file stored in %s", storePath)


class networkToRun:
    def __init__(self,
                 AUDIO_LSTM_HIDDEN_LIST=[256, 256], audio_dataset="TCDTIMIT", nbMFCCs=39, audio_bidirectional=True, audio_features='conv',
                 CNN_NETWORK="google",
                 cnn_features="conv",
                 LIP_RNN_HIDDEN_LIST=[256, 256],
                 lipRNN_bidirectional = True, lipRNN_features = 'rawRNNfeatures',
                 DENSE_HIDDEN_LIST=[512,512,512], combinationType='FC',
                 runType="combined",
                 LR_start=0.001,
                 allowSubnetTraining=False,
                 forceTrain=False,
                 overwriteSubnets = False,
                 dataset="TCDTIMIT", test_dataset=None,
                 addNoisyAudio = False):
        # Audio
        self.AUDIO_LSTM_HIDDEN_LIST = AUDIO_LSTM_HIDDEN_LIST    # LSTM architecture for audio part
        self.audio_dataset = audio_dataset  # training here only works for TCDTIMIT at the moment; for that go to audioSR/RNN.py. This variable is used to get the stored results from that python script
        self.nbMFCCs = nbMFCCs
        self.audio_bidirectional = audio_bidirectional
        self.audio_features = audio_features  # output features of audio network: raw LSTM outputs or preclassified before giving to combination FC net?

        # Lipreading
        self.CNN_NETWORK            = CNN_NETWORK               # only "google" for now. Could also use resnet50 or cifar10 from lipreading/buildNetworks.py
        self.cnn_features           = cnn_features              # conv or dense: output features of CNN that are passed on. For a CNN combined network, it's passed to the concat layer. For a CNN-LSTM network, it's the features passed to the LSTM lipreading layers
                                                                # conv -> output is 512x7x7=25.088 features -> huge combination FC networks. Performance is better though
        self.LIP_RNN_HIDDEN_LIST    = LIP_RNN_HIDDEN_LIST       # LSTM network on top of the lipreading CNNs
        self.lipRNN_bidirectional   = lipRNN_bidirectional
        self.lipRNN_features        = lipRNN_features

        # Combination network
        self.DENSE_HIDDEN_LIST      = DENSE_HIDDEN_LIST         # dense layers for combining audio and lipreading networks
        self.combinationType        = combinationType

        # Others
        self.runType                = runType       # audio, lipreading or combined
        self.LR_start               = LR_start
        self.allowSubnetTraining    = allowSubnetTraining   # eg for first time CNN-LSTM training, you need to fix the pretrained CNN parameters or they will be lost during training of the whole network
                                                            # only set this to true for a second round of training on a network that has already been trained, or on combined networks.
        self.forceTrain             = forceTrain   # If False, just test the network outputs when the network already exists.
                                              # If forceTrain == True, train it anyway before testing
                                              # If True, set the LR_start low enough so you don't move too far out of the objective minimum
        self.overwriteSubnets       = overwriteSubnets  #overwrite subnets with seperatly trained lipreading/audio parts. Useful if you've managed to improve a subnet and don't want to retrain the whole combined net from scratch

        # For audio, there are different datasets we can use (TIMIT, TCDTIMIT, combined)
        # this only works for getting the results from the _trainInfo.pkl files for now, not for actually training those networks from here
        # to do that, see audioSR/RNN.py
        self.dataset = dataset
        if test_dataset == None: self.test_dataset = self.dataset  # if only "TCDTIMIT", it's the lipspeakers. "TCDTIMITvolunteers" is the volunteers
        else: self.test_dataset = test_dataset

        # datasetType (of the training set: 'lipspeakers' or 'volunteers') is used in determining the correct path for this model
        if "TCDTIMIT" in dataset:
            if "volunteers" in dataset:     self.datasetType = "volunteers"
            else:                           self.datasetType = "lipspeakers"
        else:
            self.datasetType = ""

        self.addNoisyAudio = addNoisyAudio   #append audio data that contins noise to training set


def getManyNetworkResults(networks, resultsType="unknownResults", roundParams=False, withNoise=False,
                          noiseType='white', ratio_dB=0):
    results = {'resultsType': resultsType}
    results['audio'] = {}
    results['lipreading'] = {}
    results['combined'] = {}

    failures = []

    for networkParams in tqdm(networks, total=len(networks)):
        logger_combined.info("\n\n\n\n ################################")
        logger_combined.info("Getting results from network...")
        logger_combined.info("Network properties: ")
        #pprint(vars(networkParams))
        try:
            if networkParams.forceTrain == True:
                runManyNetworks([networkParams])
            thisResults = {}
            thisResults['values'] = []
            thisResults['dataset'] = networkParams.dataset
            thisResults['test_dataset'] = networkParams.test_dataset
            thisResults['audio_dataset'] = networkParams.audio_dataset

            model_paths, model_save, networkName = getModelPaths(AUDIO_LSTM_HIDDEN_LIST=networkParams.AUDIO_LSTM_HIDDEN_LIST, audio_features=networkParams.audio_features, audio_bidirectional=networkParams.audio_bidirectional,
                                                    CNN_NETWORK=networkParams.CNN_NETWORK,
                                                    DENSE_HIDDEN_LIST=networkParams.DENSE_HIDDEN_LIST, combinationType=networkParams.combinationType,
                                                    LIP_RNN_HIDDEN_LIST=networkParams.LIP_RNN_HIDDEN_LIST, lipRNN_features=networkParams.lipRNN_features,
                                                    allowSubnetTraining=networkParams.allowSubnetTraining,
                                                    cnn_features=networkParams.cnn_features, dataset=networkParams.dataset,  datasetType=networkParams.datasetType, addNoisyAudio=networkParams.addNoisyAudio,
                                                    lipRNN_bidirectional=networkParams.lipRNN_bidirectional,
                                                    nbMFCCs=networkParams.nbMFCCs, nbPhonemes=39, runType=networkParams.runType,
                                                    audio_dataset=networkParams.audio_dataset, getName=True)
            logger_combined.info("Getting results for %s", model_save)

            network_train_info = getNetworkResults(model_save)
            if network_train_info==-1:
                raise IOError("this model doesn't have any stored results")
            #import pdb;pdb.set_trace()

            # audio networks can be run on TIMIT or combined as well
            if networkParams.runType == 'audio' and networkParams.audio_dataset != networkParams.test_dataset:
                testType = "_" + networkParams.test_dataset
            elif networkParams.runType != 'audio' and networkParams.test_dataset != networkParams.dataset:
                testType = "_" + networkParams.test_dataset
            else:    testType = ""
            if roundParams:
                testType = "_roundParams" + testType

            if networkParams.runType != 'lipreading' and withNoise:
                thisResults['values'] = [
                    network_train_info['final_test_cost_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType],
                    network_train_info['final_test_acc_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType],
                    network_train_info['final_test_top3_acc_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType]]
            else:
                try: val_acc = max(network_train_info['val_acc'])
                except:
                    try: val_acc = max(network_train_info['test_acc'])
                    except: val_acc = network_train_info['final_test_acc']
                thisResults['values'] = [network_train_info['final_test_cost' + testType],
                               network_train_info['final_test_acc' + testType],
                               network_train_info['final_test_top3_acc' + testType], val_acc]

            thisResults['nb_params'] = network_train_info['nb_params']
            thisResults['niceName'] = networkName

            # eg results['audio']['2Layer_256_256_TIMIT'] = [0.8, 79.5, 92,6]  #test cost, test acc, test top3 acc
            results[networkParams.runType][model_save] = thisResults

        except:
            logger_combined.info('caught this error: ' + traceback.format_exc());
            # import pdb;pdb.set_trace()
            failures.append(networkParams)

    logger_combined.info("\n\nDONE getting stored results from networks")
    logger_combined.info("####################################################")

    if len(failures) > 0:
        logger_combined.info("Couldn't get %s results from %s networks...", resultsType, len(failures))
        for failure in failures:
            pprint(vars(failure))
        if autoTrain or query_yes_no("\nWould you like to evalute the networks now?\n\n"):
            logger_combined.info("Running networks...")
            runManyNetworks(failures, withNoise=withNoise,noiseType=noiseType,ratio_dB=ratio_dB)
            mainGetResults(failures, withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB)

        logger_combined.info("Done training.\n\n")
        #import pdb; pdb.set_trace()
    return results


def getNetworkResults(save_name, logger=logger_combined): #copy-pasted from loadPreviousResults
    if os.path.exists(save_name + ".npz") and os.path.exists(save_name + "_trainInfo.pkl"):
        old_train_info = unpickle(save_name + '_trainInfo.pkl')
        #import pdb;pdb.set_trace()
        if type(old_train_info) == dict:  # normal case
            network_train_info = old_train_info  # load old train info so it won't get lost on retrain

            if not 'final_test_cost' in network_train_info.keys():
                network_train_info['final_test_cost'] = min(network_train_info['test_cost'])
            if not 'final_test_acc' in network_train_info.keys():
                network_train_info['final_test_acc'] = max(network_train_info['test_acc'])
            if not 'final_test_top3_acc' in network_train_info.keys():
                network_train_info['final_test_top3_acc'] = max(network_train_info['test_topk_acc'])
        else:
            logger.warning("old trainInfo found, but wrong format: %s", save_name + "_trainInfo.pkl")
            # do nothing
    else:
        return -1
    return network_train_info

    
# networks is a list of dictionaries, where each dictionary contains the needed parameters for training
def runManyNetworks(networks, withNoise=False,noiseType='white',ratio_dB=0):
    results = {}
    failures = []
    if justTest:
        logger_combined.warning("\n\n!!!!!!!!! WARNING !!!!!!!!!!   \n justTest = True")
        if not query_yes_no("\nAre you sure you want to continue?\n\n"):
            return -1
    for network in tqdm(networks,total=len(networks)):
        print("\n\n\n\n ################################")
        print("Training new network...")
        print("Network properties: ")
        pprint(vars(network))
        try:
            model_save, test_results = runNetwork(AUDIO_LSTM_HIDDEN_LIST = network.AUDIO_LSTM_HIDDEN_LIST, audio_features= network.audio_features,
                                                  audio_bidirectional    = network.audio_bidirectional,
                                                  CNN_NETWORK            = network.CNN_NETWORK,
                                                  cnn_features           = network.cnn_features,
                                                  LIP_RNN_HIDDEN_LIST    = network.LIP_RNN_HIDDEN_LIST,
                                                  lipRNN_bidirectional   = network.lipRNN_bidirectional, lipRNN_features= network.lipRNN_features,
                                                  DENSE_HIDDEN_LIST      = network.DENSE_HIDDEN_LIST, combinationType = network.combinationType,
                                                  dataset=network.dataset, datasetType = network.datasetType, test_dataset=network.test_dataset, addNoisyAudio=network.addNoisyAudio,
                                                  runType                = network.runType,
                                                  LR_start               = network.LR_start,
                                                  allowSubnetTraining    = network.allowSubnetTraining,
                                                  forceTrain             = network.forceTrain,
                                                  overwriteSubnets       = network.overwriteSubnets,
                                                  audio_dataset=network.audio_dataset,
                                                  withNoise=withNoise,noiseType=noiseType,ratio_dB=ratio_dB)
            print(model_save)
            name = model_save + ("_Noise" +noiseType+"_"+str(ratio_dB) if withNoise else "")
            results[name] = test_results #should be test_cost, test_acc, test_topk_acc

        except:
            print('caught this error: ' + traceback.format_exc());
            #import pdb;            pdb.set_trace()

            failures.append(network)
    print("#########################################################")
    print("\n\n\n DONE running all networks")

    if len(failures) > 0:
        print("Some networks failed to run...")
        #import pdb;pdb.set_trace()
    return results

def runNetwork(AUDIO_LSTM_HIDDEN_LIST, audio_features, audio_bidirectional, CNN_NETWORK, cnn_features, lipRNN_bidirectional, LIP_RNN_HIDDEN_LIST, lipRNN_features, DENSE_HIDDEN_LIST, combinationType,
               dataset, datasetType, test_dataset, addNoisyAudio, runType, LR_start, allowSubnetTraining, overwriteSubnets, forceTrain, audio_dataset, withNoise=False, noiseType='white', ratio_dB=0):

    batch_size_audio = 1  #only works processing 1 video at a time. The lipreading CNN then processes as a batch all the images in this video
    max_num_epochs = 20
    
    nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
    nbPhonemes = 39  # number output neurons

    logger_combined.info("LR_start = %s", str(LR_start))
    LR_decay= 0.5#0.7071
    logger_combined.info("LR_decay = %s", str(LR_decay))
    

    # get paths of subnetworks, save in model_paths.
    # path to save this network = model_save.

    model_paths, model_save = getModelPaths(AUDIO_LSTM_HIDDEN_LIST, audio_features,CNN_NETWORK, DENSE_HIDDEN_LIST, combinationType,
                                                       LIP_RNN_HIDDEN_LIST, lipRNN_features, allowSubnetTraining, audio_bidirectional,
                                                       cnn_features, dataset, datasetType, addNoisyAudio, lipRNN_bidirectional,
                                                       nbMFCCs, nbPhonemes, runType, audio_dataset)
    if not os.path.exists(os.path.dirname(model_save)): os.makedirs(os.path.dirname(model_save))
    # log file
    if logToFile:
        if ".npz" in model_save: logFile = model_save.replace(".npz",'.log')
        else: logFile = model_save + ".log"
        if os.path.exists(logFile):
            fh = logging.FileHandler(logFile)       # append to existing log
        else:
            fh = logging.FileHandler(logFile, 'w')  # create new logFile
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger_combined.addHandler(fh)

        print("log file: ", logFile)
    #############################################################

    
    logger_combined.info("\n\n\n\n STARTING NEW EVALUATION/TRAINING SESSION AT " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    
    ##### IMPORTING DATA #####
    # Set locations for DATA, LOG, PARAMETERS, TRAIN info
    if "TCDTIMIT" in dataset: dataset = "TCDTIMIT" #so it works for both volunteers and lipspeakers
    root_dir = os.path.expanduser('~/TCDTIMIT/combinedSR/' + dataset)
    database_binaryDir = root_dir + '/binary'
    processedDir = database_binaryDir + "_finalProcessed"

    # lipspeakers: load all in mem at start; volunteers -> not possible, so load per speaker
    # if datasetType == "lipspeakers":
    #     loadPerSpeaker = False
    # else: loadPerSpeaker = True
    if "volunteers" in test_dataset:
        loadPerSpeaker = True
    else:
        loadPerSpeaker = False


    logger_combined.info('  data source: ' + database_binaryDir)
    storeProcessed = True  # if you have about 10GB hdd space, you can increase the speed by not reprocessing it each iteration
    # you can just run this program and it will generate the files the first time it encounters them, or generate them manually with datasetToPkl.py

    # some data for debugging
    debugFunctions = False; debugData = None
    if debugFunctions:
        debugData, datasetFiles, testSpeakerFiles = getData(database_binaryDir, datasetType, processedDir, getDebugData=True)
    else:
        datasetFiles, testSpeakerFiles = getData(database_binaryDir, datasetType, processedDir, getDebugData=False)

    ##### BUIDING MODEL #####
    logger_combined.info('\n\n* Building network ...')
    network = NeuralNetwork('combined', data=debugData, loadPerSpeaker = loadPerSpeaker, test_dataset=test_dataset,
                            num_features=nbMFCCs, lstm_hidden_list=AUDIO_LSTM_HIDDEN_LIST, audio_features=audio_features,
                            num_output_units=nbPhonemes, bidirectional=audio_bidirectional,
                            cnn_network=CNN_NETWORK, cnn_features = cnn_features,
                            lipRNN_hidden_list=LIP_RNN_HIDDEN_LIST, lipRNN_bidirectional=lipRNN_bidirectional,
                            lipRNN_features=lipRNN_features, dense_hidden_list=DENSE_HIDDEN_LIST, combinationType=combinationType,
                            model_paths=model_paths, save_name=model_save,
                            debug=False)


    # get the name of the model we're training/evaluating
    logger_combined.info(' Network built. \n\nTrying to load stored model: %s', runType)

    # Try to load stored model
    success = network.setNetworkParams(runType, overwriteSubnets=overwriteSubnets)

    ##### COMPILING FUNCTIONS #####
    logger_combined.info("\n\n* Compiling functions ...")
    network.build_functions(runType=runType, train=True, debug=False,
                            allowSubnetTraining=allowSubnetTraining)

    # if runType model already exists (and loaded successfully), just TEST it.

    if success and not forceTrain:
        if ROUND_PARAMS: #safety for if we forget to set round_params to false when training
            logger_combined.info("Loading Rounded Parameters...")
            network.setNetworkParams(runType, roundParams=ROUND_PARAMS, overwriteSubnets=overwriteSubnets)

        logger_combined.info("\n\n* Evaluating Test set ...")

        if withNoise:  #TODO a bit of a hack
            noiseTypes = ['white', 'voices']
            ratio_dBs = [0, -3, -5, -10]
            for noiseType in noiseTypes:
                for ratio_dB in ratio_dBs:
                    logger_combined.info("Noise type: %s , %s", noiseType, ratio_dB)
                    testResults = network.finalNetworkEvaluation(save_name=model_save,
                                                                 database_binaryDir=database_binaryDir,
                                                                 processedDir=processedDir, runType=runType,
                                                                 storeProcessed=storeProcessed,
                                                                 testSpeakerFiles=testSpeakerFiles, withPreds=getConfusionMatrix,
                                                                 withNoise = withNoise, noiseType = noiseType, ratio_dB = ratio_dB,
                                                                 roundParams=ROUND_PARAMS)
        else:
            testResults = network.finalNetworkEvaluation(save_name=model_save,
                                                         database_binaryDir=database_binaryDir, viseme=viseme,
                                                         processedDir=processedDir, runType=runType,
                                                         storeProcessed=storeProcessed,
                                                         testSpeakerFiles=testSpeakerFiles, withPreds=getConfusionMatrix,
                                                         withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB,
                                                         roundParams=ROUND_PARAMS)

    else: # network doesn't exist, we need to train it first. Or we forced training
        # if we loaded an existing network and force training, make LRsmaller so we don't lose the benefits of our pretrained network
        if forceTrain and success: LR_start = LR_start/10.0
        ##### TRAINING #####
        logger_combined.info("\n\n* Training ...")


        testResults = network.train(datasetFiles, database_binaryDir=database_binaryDir, runType=runType,
                      storeProcessed=True, processedDir=processedDir, viseme=viseme,
                      num_epochs=max_num_epochs,
                      batch_size=batch_size_audio, LR_start=LR_start, LR_decay=LR_decay,
                      compute_confusion=False, debug=False, justTest=justTest,
                      withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB, addNoisyAudio=addNoisyAudio,
                      save_name=model_save)

    logger_combined.info("\n\n* Done")
    logger_combined.info('Total time: {:.3f}'.format(time.time() - program_start_time))

    # close the log file handler to be able to log to new file
    if logToFile:
        fh.close()
        logger_combined.removeHandler(fh)
    return model_save, testResults #so you know which network has been trained

def getData(database_binaryDir, datasetType, processedDir, getDebugData=False):
    # just get the names
    testVolunteerNumbers = ["13F", "15F", "21M", "23M", "24M", "25M", "28M", "29M", "30F", "31F", "34M", "36F", "37F",
                            "43F", "47M", "51F", "54M"];
    testVolunteers = [str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers];
    lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
    allSpeakers = [f for f in os.listdir(database_binaryDir) if
                   os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"]
    trainVolunteers = [f for f in allSpeakers if not (f in testVolunteers or f in lipspeakers)];
    if datasetType == "combined":
        trainingSpeakerFiles = trainVolunteers + lipspeakers
        testSpeakerFiles = testVolunteers
    else:  # datasetType == "volunteers":
        trainingSpeakerFiles = trainVolunteers
        testSpeakerFiles = testVolunteers

    datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]  #lipspeakers are loaded all together in combinedNN_tools.py
    # get a sample of the dataset to debug the network
    if getDebugData:
        if datasetType == "lipspeakers":
            lipspkr_path = os.path.expanduser("~/TCDTIMIT/combinedSR/TCDTIMIT/binaryLipspeakers/allLipspeakersTest.pkl")
            logger_combined.info("data: lipspkr_path")
            debugData = unpickle(lipspkr_path)
        else:
            debugData, _, _ = preprocessingCombined.getOneSpeaker(trainingSpeakerFiles[0],
                                                             sourceDataDir=database_binaryDir,
                                                             storeProcessed=True,
                                                             processedDir=processedDir,
                                                             trainFraction=1.0, validFraction=0.0,
                                                             verbose=False)

        return debugData, datasetFiles, testSpeakerFiles
    else:
        return datasetFiles, testSpeakerFiles


def getModelPaths(AUDIO_LSTM_HIDDEN_LIST, audio_features, CNN_NETWORK, DENSE_HIDDEN_LIST, combinationType, LIP_RNN_HIDDEN_LIST, lipRNN_features, allowSubnetTraining,
                  audio_bidirectional, cnn_features, dataset, datasetType, addNoisyAudio, lipRNN_bidirectional, nbMFCCs,
                  nbPhonemes, runType, audio_dataset="combined", getName = False): #combined audio dataset means both TIMIT and TCDTIMIT data
    # ugly hack for volunteers
    if "TCDTIMIT" in dataset:
        dataset = "TCDTIMIT"
    model_paths = {}

    # AUDIO
    #audio_dataset = "combined" # = TCDTIMIT + TIMIT datasets
    audio_model_name = str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
            [str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) + "_nbMFCC" + str(nbMFCCs) + \
                       ("_bidirectional" if audio_bidirectional else "_unidirectional") + "_" + audio_dataset
    audio_model_dir = os.path.expanduser("~/TCDTIMIT/audioSR/" + audio_dataset + "/results")
    model_paths['audio'] = os.path.join(audio_model_dir, audio_model_name + ".npz")
    audioNiceName = "Audio:" + ' '.join(
            ["LSTM", str(AUDIO_LSTM_HIDDEN_LIST[0]), "/", str(len(AUDIO_LSTM_HIDDEN_LIST)), ("PF" if audio_features=='dense' else 'RF')])


    # LIPREADING
    lip_model_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset + "/results/CNN"))
    network_type = CNN_NETWORK
    lip_CNN_model_name = (datasetType + "_" if datasetType != "" else "") + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbPhonemes)
    model_paths['CNN'] = os.path.join(lip_model_dir, lip_CNN_model_name + ".npz")
    lipNiceName = "Lipreading: "+ CNN_NETWORK + " " + ("viseme" if viseme else "phoneme")

    # for binary CNN networks
    if "binary" in CNN_NETWORK: #not implemented yet
        lip_model_dir += "_binaryNet"
        lip_CNN_model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbPhonemes) + "_binary"
        model_paths['CNN'] = os.path.join(lip_model_dir, lip_CNN_model_name +".npz")

    # for CNN-LSTM networks
    elif LIP_RNN_HIDDEN_LIST != None:
        lip_model_dir += "_LSTM"
        lip_CNN_LSTM_model_name = lip_CNN_model_name + "_LSTM" + (
        "_bidirectional" if lipRNN_bidirectional else "_unidirectional") \
                                  + "_" + '_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST]) + "_" + cnn_features \
                                  + ("_viseme" if viseme else "")
        model_paths['CNN_LSTM'] = os.path.join(lip_model_dir, lip_CNN_LSTM_model_name + ".npz")
        lipNiceName = "Lipreading: " + ' '.join([CNN_NETWORK, ("PF" if cnn_features == 'dense' else 'RF'),
                                "+ LSTM",str(LIP_RNN_HIDDEN_LIST[0]),"/",str(len(LIP_RNN_HIDDEN_LIST)), ('PC' if lipRNN_features=='dense' else 'RF')])


    # COMBINED (audio+lip+combining)
    root_dir = os.path.expanduser('~/TCDTIMIT/combinedSR/' + dataset)
    store_dir = root_dir + os.sep + "results" + os.sep + ("CNN_LSTM" if LIP_RNN_HIDDEN_LIST != None else "CNN") + (
    os.sep + datasetType if datasetType != "" else "")
    if not os.path.exists(store_dir): os.makedirs(store_dir)

    model_name = "RNN__" + str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
            [str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) \
                 + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if audio_bidirectional else "_unidirectional") \
                 + ("_audioDense" if audio_features == 'dense' else "") \
                 + "__" + "CNN_" + CNN_NETWORK + "_" + cnn_features \
                 + ("_lipRNN_" if LIP_RNN_HIDDEN_LIST != None else "") + (
                     '_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST]) if LIP_RNN_HIDDEN_LIST != None else "") \
                 + ("_RNNfeaturesDense" if lipRNN_features == 'dense' else "") \
                 + ("__allowSubnetTraining" if allowSubnetTraining else "") \
                 + ("__FC_" + '_'.join([str(layer) for layer in DENSE_HIDDEN_LIST]) if combinationType == 'FC' else "") \
                 + ("__attention" if combinationType == 'attention' else "") \
                 + "__" + dataset + ("_" + datasetType if datasetType != "" else "") \
                 + ("__withNoisyAudio" if addNoisyAudio else "")

    ## TODO: Ugly hack for attention networks
    # model_name = 'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional_audioDense__CNN_google_conv_lipRNN_256_256_RNNfeaturesDense__attention__TCDTIMIT_lipspeakers__withNoisyAudio'
    # model_name = 'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional__CNN_google_conv_lipRNN_256_256__attention__TCDTIMIT_lipspeakers__retrainedWithNoisyAudioWhite'
    # model_name = 'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional__CNN_google_conv_lipRNN_256_256__attention__TCDTIMIT_lipspeakers__retrainedWithNoisyAudio'
    model_name = 'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional__CNN_google_conv_lipRNN_256_256__attention__TCDTIMIT_lipspeakers__retrainedWithNoisyAudio_allowSubnetTraining'

    model_paths['combined'] = os.path.join(store_dir, model_name + ".npz")

    combinationNiceName = ("FC " + str(DENSE_HIDDEN_LIST[0]) + " / " + str(len(DENSE_HIDDEN_LIST)) if combinationType == 'FC' else "")\
                + (' attention' if combinationType == 'attention'  else "")
    combinedNiceName = "Multimodal: \n" + " ".join([audioNiceName,"|", lipNiceName, "| Combination:", combinationNiceName,
                                 ("EndToEnd" if allowSubnetTraining else "")])

    # set correct paths for storage of results
    if runType == 'audio':
        model_save = model_paths['audio']
        store_dir = audio_model_dir
        niceName = audioNiceName
    elif runType == 'lipreading':
        store_dir = lip_model_dir
        niceName = lipNiceName
        if LIP_RNN_HIDDEN_LIST != None:
            model_save = model_paths['CNN_LSTM']
        else:
            model_save = model_paths['CNN']
    elif runType == 'combined':
        model_save = model_paths['combined']
        niceName = combinedNiceName
    else:
        raise IOError("can't save network params; network type not found")
    model_save = model_save.replace(".npz", "")

    if getName:
        return model_paths, model_save, niceName
    else:
        return model_paths, model_save



if __name__ == "__main__":
    main()