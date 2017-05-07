from __future__ import print_function

import warnings
from time import gmtime, strftime
from pprint import pprint  #printing properties of networkToTrain objects
#pprint(vars(a))

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting


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
import preprocessingCombined

#############################################################

logToFile = True

def main():
    # each element needs AUDIO_LSTM_HIDDEN_LIST, CNN_NETWORK, cnn_features, LIP_RNN_HIDDEN_LIST, DENSE_HIDDEN_LIST, datasetType, runType
    print("starting training of many networks...")
    networkList = [
        #
        # # # # # ### LIPREADING ###
        # # # # # # # # CNN
        # networkToTrain(runType="lipreading",LIP_RNN_HIDDEN_LIST=None),
        # # # # # #
        # # # # # # # CNN-LSTM -> by default only the LSTM part is trained (line 713 in build_functions in combinedNN_tools.py
        # # # # # # #          -> you can train everything (also CNN parameters), but only do this after the CNN-LSTM has trained with fixed CNN parameters, otherwise you'll move far out of your optimal point
        # networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[64]),
        # # # # #
        # networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256]),
        # #
        # networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[512,512]),
        # #
        # networkToTrain(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[64]),
        # networkToTrain(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256]),
        #
        # # # # # ### AUDIO ###  -> see audioSR/RNN.py, there it can run in batch mode which is much faster
        # networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64]),#,forceTrain=True),
        # networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256]),#, forceTrain=True),
        # networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512]),#,forceTrain=True),
        #
        # # # ### COMBINED ###
        # # # # lipspeakers
        # # # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[128, 64, 64]),
        # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # networkToTrain(cnn_features="conv", LIP_RNN_HIDDEN_LIST=None,
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256,256],
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        #
        # ## conv
        # networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                cnn_features="conv", LIP_RNN_HIDDEN_LIST=[64],
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                cnn_features="conv", LIP_RNN_HIDDEN_LIST=[512, 512],
        #                DENSE_HIDDEN_LIST=[512, 512, 512]),
        # #
        # # # TODO: train with params CNN and audio made trainable
        # # networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        # #                cnn_features="conv", LIP_RNN_HIDDEN_LIST=[64],
        # #                DENSE_HIDDEN_LIST=[512, 512, 512], allowSubnetTraining=True),


        # # volunteers
        # ## lipreading
        # networkToTrain(runType="lipreading", LIP_RNN_HIDDEN_LIST=None,
        #                datasetType="volunteers"),
        # networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[64],
        #                datasetType="volunteers"),
        # networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[512,512],
        #                datasetType="volunteers"),

        # audio -> RNN.py
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                        datasetType="volunteers"),
        # networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                datasetType="volunteers"),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256]),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512])

        # # Combined
        # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[512, 512, 512],
        #                datasetType="volunteers"),
        # networkToTrain(cnn_features="conv",  LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[512, 512, 512],
        #                datasetType="volunteers"),
        #
        # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[64], DENSE_HIDDEN_LIST=[512, 512, 512],
        #                datasetType="volunteers"),
        # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256,256], DENSE_HIDDEN_LIST=[512, 512, 512],
        #                datasetType="volunteers")#, forceTrain=True),
    ]
    trainManyNetworks(networkList)

class networkToTrain:
    def __init__(self,
                 AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                 CNN_NETWORK="google",
                 cnn_features="dense",
                 LIP_RNN_HIDDEN_LIST=[256, 256],
                 lipRNN_bidirectional = True,
                 DENSE_HIDDEN_LIST=[512,512,512],
                 datasetType="lipspeakers",
                 runType="combined",
                 LR_start=0.001,
                 allowSubnetTraining=False,
                 forceTrain=False):
        self.AUDIO_LSTM_HIDDEN_LIST = AUDIO_LSTM_HIDDEN_LIST    # LSTM architecture for audio part
        self.CNN_NETWORK            = CNN_NETWORK               # only "google" for now. Could also use resnet50 or cifar10 from lipreading/buildNetworks.py
        self.cnn_features           = cnn_features              # conv or dense: output features of CNN that are passed on. For a CNN combined network, it's passed to the concat layer. For a CNN-LSTM network, it's the features passed to the LSTM lipreading layers
                                                                # conv -> output is 512x7x7=25.088 features -> huge combination FC networks. Performance is better though
        self.LIP_RNN_HIDDEN_LIST    = LIP_RNN_HIDDEN_LIST       # LSTM network on top of the lipreading CNNs
        self.lipRNN_bidirectional   = lipRNN_bidirectional
        self.DENSE_HIDDEN_LIST      = DENSE_HIDDEN_LIST         # dense layers for combining audio and lipreading networks
        self.datasetType            = datasetType   # volunteers or lipreaders
        self.runType                = runType       # audio, lipreading or combined
        self.LR_start               = LR_start
        self.allowSubnetTraining    = allowSubnetTraining   # eg for first time CNN-LSTM training, you need to fix the pretrained CNN parameters or they will be lost during training of the whole network
                                                            # only set this to true for a second round of training on a network that has already been trained, or on combined networks.
        self.forceTrain             = forceTrain   # If False, just test the network outputs when the network already exists.
                                              # If forceTrain == True, train it anyway before testing
                                              # If True, set the LR_start low enough so you don't move too far out of the objective minimum


# networks is a list of dictionaries, where each dictionary contains the needed parameters for training
def trainManyNetworks(networks):
    failures = []
    for network in tqdm(networks,total=len(networks)):
        print("\n\n\n\n ################################")
        print("Training new network...")
        print("Network properties: ")
        pprint(vars(network))
        try:
            trainNetwork(AUDIO_LSTM_HIDDEN_LIST     = network.AUDIO_LSTM_HIDDEN_LIST,
                         CNN_NETWORK            = network.CNN_NETWORK,
                         cnn_features           = network.cnn_features,
                         LIP_RNN_HIDDEN_LIST    = network.LIP_RNN_HIDDEN_LIST,
                         lipRNN_bidirectional   = network.lipRNN_bidirectional,
                         DENSE_HIDDEN_LIST      = network.DENSE_HIDDEN_LIST,
                         datasetType            = network.datasetType,
                         runType                = network.runType,
                         LR_start               = network.LR_start,
                         allowSubnetTraining    = network.allowSubnetTraining,
                         forceTrain             = network.forceTrain)
        except:
            name = network.runType + "_" + '_'.join([str(layer) for layer in network.AUDIO_LSTM_HIDDEN_LIST]) \
                   + network.cnn_features + '_'.join([str(layer) for layer in network.LIP_RNN_HIDDEN_LIST]) \
                    + '_'.join([str(layer) for layer in network.DENSE_HIDDEN_LIST]) + "_" + network.datasetType
            failures.append(name)
    print("#########################################################")
    print("\n\n\n DONE training all networks")

    if len(failures) > 0:
        print("Some networks failed to train...")
        import pdb;pdb.set_trace()
        

def trainNetwork(AUDIO_LSTM_HIDDEN_LIST, CNN_NETWORK, cnn_features,lipRNN_bidirectional, LIP_RNN_HIDDEN_LIST, DENSE_HIDDEN_LIST,
                 datasetType, runType, LR_start, allowSubnetTraining, forceTrain):
    ##### SCRIPT META VARIABLES #####
    VERBOSE = True
    compute_confusion = False  # TODO: ATM this is not implemented
    
    batch_size_audio = 1  #only works processing 1 video at a time. The lipreading CNN then processes as a batch all the images in this video
    max_num_epochs = 20
    
    nbMFCCs = 39 # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
    nbPhonemes = 39  # number output neurons
    #AUDIO_LSTM_HIDDEN_LIST = [256, 256]
    audio_bidirectional = True
    
    # # lipreading
    # CNN_NETWORK = "google"
    # # using CNN-LSTM combo: what to input to LSTM? direct conv outputs or first through dense layers?
    # cnn_features = 'conv' #'dense' # 39 outputs as input to LSTM
    # LIP_RNN_HIDDEN_LIST = None #[256,256]  # set to None to disable CNN-LSTM architecture
    #
    # # after concatenation of audio and lipreading, which dense layers before softmax?
    # DENSE_HIDDEN_LIST = [128,64,64] #[128,128,128,128]
    #
    # # Decaying LR
    # LR_start = 0.001
    logger_combined.info("LR_start = %s", str(LR_start))
    LR_fin = 0.0000001
    # logger_combined.info("LR_fin = %s", str(LR_fin))
    #LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
    LR_decay= 0.5#0.7071
    logger_combined.info("LR_decay = %s", str(LR_decay))
    
    # Set locations for DATA, LOG, PARAMETERS, TRAIN info
    dataset = "TCDTIMIT"
    root_dir = os.path.expanduser('~/TCDTIMIT/combinedSR/' + dataset)
    database_binaryDir = root_dir + '/binary'
    processedDir = database_binaryDir + "_finalProcessed"
    
    # datasetType = "lipspeakers"  # ""volunteers";
    if datasetType == "lipspeakers": loadPerSpeaker = False
    else: loadPerSpeaker = True
    
    store_dir = root_dir + os.sep + "results" + os.sep + ("CNN_LSTM" if LIP_RNN_HIDDEN_LIST != None else "CNN") + os.sep + datasetType
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    
    # # which part of the network to train/save/...
    # # runType = 'audio'
    # # runType = 'lipreading'
    # runType = 'combined'
    ###########################
    
    model_paths = {}
    # audio network + cnnNetwork + classifierNetwork
    model_name = "RNN__" + str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join([str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) \
                         + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if audio_bidirectional else "_unidirectional") +  "__" \
                 + "CNN_" + CNN_NETWORK + "_" + cnn_features \
                 + ("_lipRNN_" if LIP_RNN_HIDDEN_LIST != None else "") + ('_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST]) if LIP_RNN_HIDDEN_LIST != None else "") \
                 + ("_allowSubnetTraining" if allowSubnetTraining else "") + "__" \
                 + "FC_" + '_'.join([str(layer) for layer in DENSE_HIDDEN_LIST]) + "__" \
                 + dataset + "_" + datasetType
    model_paths['combined'] = os.path.join(store_dir, model_name + ".npz")

    # for loading stored audio models
    audio_dataset = "combined" #""combined" # TCDTIMIT + TIMIT datasets
    audio_model_name = str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
            [str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) + "_nbMFCC" + str(nbMFCCs) + \
                       ("_bidirectional" if audio_bidirectional else "_unidirectional") + "_" + audio_dataset
    audio_model_dir = os.path.expanduser("~/TCDTIMIT/audioSR/"+audio_dataset+"/results")
    model_paths['audio'] = os.path.join(audio_model_dir, audio_model_name + ".npz")
    
    # for loading stored lipreading models
    lip_model_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset + "/results/CNN"))
    viseme = False; network_type = CNN_NETWORK
    lip_CNN_model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbPhonemes)
    model_paths['CNN'] = os.path.join(lip_model_dir, lip_CNN_model_name + ".npz")
    
    # for CNN-LSTM networks
    if LIP_RNN_HIDDEN_LIST != None:
        lip_model_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset + "/results/CNN_LSTM"))
        lip_CNN_LSTM_model_name = lip_CNN_model_name + "_LSTM" + ("_bidirectional" if lipRNN_bidirectional else "_unidirectional") \
                                  + "_" + '_'.join([str(layer) for layer in LIP_RNN_HIDDEN_LIST]) + "_" + cnn_features
        model_paths['CNN_LSTM'] = os.path.join(lip_model_dir, lip_CNN_LSTM_model_name + ".npz")


    # set correct paths for storage of results
    if runType == 'audio':
        model_save = model_paths['audio']
        store_dir = audio_model_dir
    elif runType == 'lipreading':
        store_dir = lip_model_dir
        if LIP_RNN_HIDDEN_LIST != None: model_save = model_paths['CNN_LSTM']
        else: model_save = model_paths['CNN']
    elif runType == 'combined':
        model_save = model_paths['combined']
    else:
        raise IOError("can't save network params; network type not found")
    model_save = model_save.replace(".npz", "")


    # log file
    if logToFile:
        logFile = store_dir + os.sep + os.path.basename(model_save) + '.log'
        if os.path.exists(logFile):
            fh = logging.FileHandler(logFile)       # append to existing log
        else:
            fh = logging.FileHandler(logFile, 'w')  # create new logFile
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger_combined.addHandler(fh)

        print("log file: ", logFile)
    #############################################################
    
    
    logger_combinedtools.info("\n\n\n\n STARTING NEW EVALUATION/TRAINING SESSION AT " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    
    ##### IMPORTING DATA #####
    logger_combined.info('  data source: ' + database_binaryDir)
    storeProcessed = True  # if you have about 10GB hdd space, you can increase the speed by not reprocessing it each iteration
    # you can just run this program and it will generate the files the first time it encounters them, or generate them manually with datasetToPkl.py

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
    else:# datasetType == "volunteers":
        trainingSpeakerFiles = trainVolunteers
        testSpeakerFiles = testVolunteers
    # else:
    #     raise Exception("invalid dataset entered")
    datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]


    # get a sample of the dataset to debug the network
    if datasetType == "lipspeakers":
        lipspkr_path = os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binaryPerVideo/allLipspeakersTest.pkl")
        data = unpickle(lipspkr_path)
    else:
        data, _, _ = preprocessingCombined.getOneSpeaker(trainingSpeakerFiles[0],
                                                             sourceDataDir=database_binaryDir,
                                                             storeProcessed=True,
                                                             processedDir=processedDir,
                                                             trainFraction=1.0, validFraction=0.0,
                                                             verbose=False)


    ##### BUIDING MODEL #####
    logger_combined.info('\n\n* Building network ...')
    network = NeuralNetwork('combined', dataset=data, loadPerSpeaker = loadPerSpeaker,
                            num_features=nbMFCCs, lstm_hidden_list=AUDIO_LSTM_HIDDEN_LIST,
                            num_output_units=nbPhonemes, bidirectional=audio_bidirectional,
                            cnn_network=CNN_NETWORK, cnn_features = cnn_features,
                            lipRNN_hidden_list=LIP_RNN_HIDDEN_LIST, lipRNN_bidirectional=lipRNN_bidirectional,
                            dense_hidden_list=DENSE_HIDDEN_LIST,
                            model_paths=model_paths,
                            debug=False)


    # get the name of the model we're training/evaluating
    logger_combined.info(' Network built. \n\nTrying to load stored model: %s', runType)

    # Try to load stored model
    success = network.setNetworkParams(runType)

    ##### COMPILING FUNCTIONS #####
    logger_combined.info("\n\n* Compiling functions ...")
    network.build_functions(runType=runType, train=True, debug=False,
                            allowSubnetTraining=allowSubnetTraining)

    # if runType model already exists (and loaded successfully), no need to train it, just evaluate.
    if success and not forceTrain:
        network.finalNetworkEvaluation(save_name=model_save,
                                       database_binaryDir=database_binaryDir,
                                       processedDir=processedDir, runType=runType,
                                       storeProcessed=storeProcessed,
                                       testSpeakerFiles=testSpeakerFiles)

    else: # network doesn't exist, we need to train it first. Or we forced training
        # if we loaded an existing network and force training, make LRsmaller so we don't lose the benefits of our pretrained network
        if forceTrain and success: LR_start = LR_start/10.0
        ##### TRAINING #####
        logger_combined.info("\n\n* Training ...")


        network.train(datasetFiles, database_binaryDir=database_binaryDir, runType=runType,
                      storeProcessed=True, processedDir=processedDir,
                      num_epochs=max_num_epochs,
                      batch_size=batch_size_audio, LR_start=LR_start, LR_decay=LR_decay,
                      compute_confusion=False, debug=False, save_name=model_save)

    logger_combined.info("\n\n* Done")
    logger_combined.info('Total time: {:.3f}'.format(time.time() - program_start_time))

    # close the log file handler to be able to log to new file
    if logToFile:
        fh.close()
        logger_combined.removeHandler(fh)

    return model_save #so you know which network has been trained



if __name__ == "__main__":
    main()