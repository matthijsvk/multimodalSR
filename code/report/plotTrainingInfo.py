import os, sys,numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pylab

fontP = FontProperties()

from general_tools import unpickle
#from combinedNN_tools import *

#############################################################
fontSize = 18
lineWidth = 3
root = os.path.expanduser('~/TCDTIMIT/results') #/results')

storeDir = os.path.expanduser(
    '~/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/multimodalSR_report/report/final/figuren/audioSR/')

### TODO: plot weight distribution of combined network -> weights of Lipreading vs Audio
# params = l_out.get_params()
# W = params[0].get_value()
# #When you print params, you will see all the parameters for l_out: [W,b]


### TODO: confusion matrices heatmaps: sns.heatmap(confusion_matrix(y_test, y_pred))

def main():
    # # plot validation accuracy through the epochs
    # network = networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=None, datasetType="lipspeakers")

    # network = networkToTrain(runType="lipreading", cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256,256])

    # network = networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256])#,  datasetType="volunteers")
    # network = networkToTrain(runType="audio",cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
    #                DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True)
    # network = networkToTrain(runType="combined",cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256,256],
    #                DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True, datasetType="volunteers")
    # network = networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256], DENSE_HIDDEN_LIST=[512, 512, 512],
    #                datasetType="volunteers", forceTrain=True)
    # plotNetworkTrainInfo(network)

    networkList = [
    #     # # # ### LIPREADING ###
    #     # # # # # # CNN
    #     networkToTrain(runType="lipreading",LIP_RNN_HIDDEN_LIST=None,forceTrain=True),
    #     # # # #
    #     # # # # # CNN-LSTM -> by default only the LSTM part is trained (line 713 in build_functions in combinedNN_tools.py
    #     # # # # #          -> you can train everything (also CNN parameters), but only do this after the CNN-LSTM has trained with fixed CNN parameters, otherwise you'll move far out of your optimal point
    #     networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[64], forceTrain=True),
    #     # # #
    #     networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],forceTrain=True),
    #
    #     networkToTrain(runType="lipreading", cnn_features="dense", LIP_RNN_HIDDEN_LIST=[512,512], forceTrain=True),
    #
    #     # # # ### AUDIO ###  -> see audioSR/RNN.py, there it can run in batch mode which is much faster
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512])
    #
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256]),
    #     networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512])

        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8,8,8]),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32,32,32]),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64,64,64]),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256,256,256]),
        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512,512,512])
    #
    #     # ### COMBINED ###
    #     # # lipspeakers
    #     networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None,
    #                    DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True),
    #     networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
    #                    DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True),
    #     networkToTrain(AUDIO_LSTM_HIDDEN_LIST=[512,512],
    #                    cnn_features="dense", LIP_RNN_HIDDEN_LIST=[256, 256],
    #                    DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True),
    #     # networkToTrain(cnn_features="conv", LIP_RNN_HIDDEN_LIST=[256, 256],
    #     #                DENSE_HIDDEN_LIST=[512, 512, 512], forceTrain=True),
    #
    #     # # volunteers
    #     # networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[512, 512, 256], datasetType="volunteers"),
    #     # networkToTrain(cnn_features="conv",  LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[512, 512, 256], datasetType="volunteers")
    ]
    # getPerfData(networkList, runType='lipreading')
    # plotPerfData(*getPerfData(networkList, runType='audio', audio_dataset='TIMIT'))
    # getPerfData(networkList, runType='combined')
    # getPerfData(networkList, runType=None)
    plotNbLayersComparison = True
    makeTableNbLayersComparison = False

    if plotNbLayersComparison:
        networkList1 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024])]

        networkList2 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024, 1024])]

        networkList3 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8, 8, 8]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32, 32, 32, 32]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64, 64, 64]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256, 256, 256]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512, 512, 512]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024,1024,1024,1024])]

        series1 = getPerfData(networkList1, runType='audio', audio_dataset='TIMIT')
        series2 = getPerfData(networkList2, runType='audio', audio_dataset='TIMIT')
        series3 = getPerfData(networkList3, runType='audio', audio_dataset='TIMIT')

        plotSeries([series1,series2,series3], colors = ['r','g','b'], labels=["One Layer", "Two Layers", "Four layers"], showNumbers=False)

    if makeTableNbLayersComparison:
        networkList1 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8]),
                    networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8]),
                    networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8,8,8]),
                    networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8, 8, 8, 8, 8, 8, 8])]
        series1 = getPerfData(networkList1, runType='audio', audio_dataset='TIMIT')
        networkList2 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32, 32]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32, 32, 32, 32])]
        series2 = getPerfData(networkList2, runType='audio', audio_dataset='TIMIT')
        networkList3 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64, 64, 64])]
        series3 = getPerfData(networkList3, runType='audio', audio_dataset='TIMIT')
        networkList4 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256, 256, 256])]
        series4 = getPerfData(networkList4, runType='audio', audio_dataset='TIMIT')
        networkList5 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512, 512, 512])]
        series5 = getPerfData(networkList5, runType='audio', audio_dataset='TIMIT')
        networkList6 = [networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024, 1024]),
                        networkToTrain(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024, 1024, 1024, 1024])]
        series6 = getPerfData(networkList6, runType='audio', audio_dataset='TIMIT')
        
        
        x, y, networkNames, runType = series1
        y = [series1[1], series2[1], series3[1],series4[1],series5[1],series6[1]]
        networkNames = ["8","32","64","256","512","1024"]
        generateTexTable("LSTMtable", ["#units/layer", "Acc"], networkNames, y)




class networkToTrain:
    def __init__(self,
                 AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                 CNN_NETWORK="google",
                 cnn_features="conv",
                 LIP_RNN_HIDDEN_LIST=[256, 256],
                 lipRNN_bidirectional=True,
                 DENSE_HIDDEN_LIST=[512, 512, 512],
                 datasetType="lipspeakers",
                 runType="combined",
                 LR_start=0.001,
                 forceTrain=False):
        self.AUDIO_LSTM_HIDDEN_LIST = AUDIO_LSTM_HIDDEN_LIST  # LSTM architecture for audio part
        self.CNN_NETWORK = CNN_NETWORK  # only "google" for now. Could also use resnet50 or cifar10 from lipreading/buildNetworks.py
        self.cnn_features = cnn_features  # conv or dense: output features of CNN that are passed on. For a CNN combinet network, it's passed to the concat layer. For a CNN-LSTM network, it's the features passed to the LSTM lipreading layers
        # conv -> output is 512x7x7=25.088 features -> huge combination FC networks. Performance is better though
        self.LIP_RNN_HIDDEN_LIST = LIP_RNN_HIDDEN_LIST  # LSTM network on top of the lipreading CNNs
        self.lipRNN_bidirectional = lipRNN_bidirectional
        self.DENSE_HIDDEN_LIST = DENSE_HIDDEN_LIST  # dense layers for combining audio and lipreading networks
        self.datasetType = datasetType  # volunteers or lipreaders
        self.runType = runType  # audio, lipreading or combined
        self.LR_start = LR_start
        self.forceTrain = forceTrain  # If False, just test the network outputs when the network already exists.
        # If forceTrain == True, train it anyway before testing
        # If True, set the LR_start low enough so you don't move too far out of the objective minimum

def getNetworkTrainInfoPath(network, audio_dataset='combined'):
    AUDIO_LSTM_HIDDEN_LIST=network.AUDIO_LSTM_HIDDEN_LIST
    CNN_NETWORK=network.CNN_NETWORK
    cnn_features=network.cnn_features
    LIP_RNN_HIDDEN_LIST=network.LIP_RNN_HIDDEN_LIST
    lipRNN_bidirectional=network.lipRNN_bidirectional
    DENSE_HIDDEN_LIST=network.DENSE_HIDDEN_LIST
    datasetType=network.datasetType
    runType=network.runType
    LR_start=network.LR_start
    forceTrain=network.forceTrain


    nbMFCCs = 39  # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
    nbPhonemes = 39  # number output neurons
    # AUDIO_LSTM_HIDDEN_LIST = [256, 256]
    audio_bidirectional = True

    # # Decaying LR
    # LR_start = 0.001
    LR_fin = 0.0000001
    # LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)  # each epoch, LR := LR * LR_decay
    LR_decay = 0.5  # 0.7071

    # Set locations for DATA, LOG, PARAMETERS, TRAIN info

    dataset = "TCDTIMIT"
    root_dir = root + "/combinedSR/" + dataset
    database_binaryDir = root_dir + '/binary'
    processedDir = database_binaryDir + "_finalProcessed"

    # datasetType = "lipspeakers"  # ""volunteers";
    if datasetType == "lipspeakers":
        loadPerSpeaker = False
    else:
        loadPerSpeaker = True

    combined_store_dir = root_dir + os.sep + "results" + os.sep + ( "CNN_LSTM" if LIP_RNN_HIDDEN_LIST != None else "CNN") + os.sep + datasetType
    if not os.path.exists(combined_store_dir): os.makedirs(combined_store_dir)

    # # which part of the network to train/save/...
    # # runType = 'audio'
    # # runType = 'lipreading'
    # runType = 'combined'
    ###########################

    model_paths = {}
    # audio network + cnnNetwork + classifierNetwork
    model_name = "RNN__" + str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
            [str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) \
                 + "_nbMFCC" + str(nbMFCCs) + (
                 "_bidirectional" if audio_bidirectional else "_unidirectional") + "__" \
                 + "CNN_" + CNN_NETWORK + "_" + cnn_features \
                 + ("_lipRNN_" if LIP_RNN_HIDDEN_LIST != None else "") + ('_'.join(
            [str(layer) for layer in LIP_RNN_HIDDEN_LIST]) if LIP_RNN_HIDDEN_LIST != None else "") + "__" \
                 + "FC_" + '_'.join([str(layer) for layer in DENSE_HIDDEN_LIST]) + "__" \
                 + dataset + "_" + datasetType
    model_paths['combined'] = os.path.join(combined_store_dir, model_name + ".npz")

    # for loading stored audio models
    # audio_dataset = "combined"  # TCDTIMIT + TIMIT datasets
    audio_model_name = str(len(AUDIO_LSTM_HIDDEN_LIST)) + "_LSTMLayer" + '_'.join(
            [str(layer) for layer in AUDIO_LSTM_HIDDEN_LIST]) + "_nbMFCC" + str(nbMFCCs) + \
                       ("_bidirectional" if audio_bidirectional else "_unidirectional") + "_" + audio_dataset
    audio_model_dir = root + "/audioSR/" + audio_dataset + "/results"
    model_paths['audio'] = os.path.join(audio_model_dir, audio_model_name + ".npz")

    # for loading stored lipreading models
    lip_model_dir = root + '/lipreading/' + dataset + "/results/CNN"
    viseme = False;
    network_type = CNN_NETWORK
    lip_CNN_model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(
        nbPhonemes)
    model_paths['CNN'] = os.path.join(lip_model_dir, lip_CNN_model_name + ".npz")

    # for CNN-LSTM networks
    if LIP_RNN_HIDDEN_LIST != None:
        lip_model_dir = os.path.join(root + '/lipreading/' + dataset + "/results/CNN_LSTM")
        lip_CNN_LSTM_model_name = lip_CNN_model_name + "_LSTM" + (
        "_bidirectional" if lipRNN_bidirectional else "_unidirectional") \
                                  + "_" + '_'.join(
                [str(layer) for layer in LIP_RNN_HIDDEN_LIST]) + "_" + cnn_features
        model_paths['CNN_LSTM'] = os.path.join(lip_model_dir, lip_CNN_LSTM_model_name + ".npz")

    # set correct paths for storage of results
    if runType == 'audio':
        model_save = model_paths['audio']
        store_dir = audio_model_dir
    elif runType == 'lipreading':
        store_dir = lip_model_dir
        if LIP_RNN_HIDDEN_LIST != None:
            model_save = model_paths['CNN_LSTM']
        else:
            model_save = model_paths['CNN']
    elif runType == 'combined':
        model_save = model_paths['combined']
    else:
        raise IOError("error; network type not found")


    model_save = model_save.replace(".npz","_trainInfo.pkl")
    return model_save


# plot train/test/validation cost, acc and top3_acc over the epochs
def plotNetworkTrainInfo(network, audio_dataset='combined'):

    # Lots of examples: http://matplotlib.org/users/pyplot_tutorial.html

    path = getNetworkTrainInfoPath(network, audio_dataset)
    print("plotting training graphs of : \n", path)
    network_train_info = unpickle(path)

    dataTypes = ['acc','topk_acc','cost']
    for dataType in dataTypes:
        plt.figure()
        ax = plt.subplot(111)

        if dataType == 'acc':
            plotYlabel = 'Top 1 Accuracy'
        elif dataType == 'topk_acc':
            plotYlabel = 'Top 3 Accuracy'
        elif dataType == 'cost':
            plotYlabel = 'Cross Entropy Cost'
        else:
            plotYlabel = 'y'

        plotXlabel = 'Epoch Number'

        one= {}; one['setType'] = 'val';
        two= {}; two['setType'] = 'test'; two['key'] = two['setType'] + "_" + dataType
        toPlot = [one, two]

        for el in toPlot:
            el['key'] = el['setType'] + "_" + dataType

            if el['setType'] == 'val': el['name'] = 'Validation Set'
            elif el['setType'] == 'test': el['name'] = 'Test Set'
            elif el['setType'] == 'train': el['name'] = 'Train Set'
            else: el['name'] = 'unknown set'

        values = [network_train_info[el['key']] for el in toPlot]
        x = range(len(values[0]))

        # red dashes, blue squares and green triangles
        line1 = ax.plot(x, values[0], 'r--', label=toPlot[0]['name'], linewidth=lineWidth)
        line2 = ax.plot(x, values[1], 'b--', label=toPlot[1]['name'], linewidth=lineWidth)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5, fontsize=fontSize)

        # ax.legend(loc='best', shadow=True, ncol=2, borderaxespad=0., prop=fontP)  # legend on bottom
        ax.set_ylabel(plotYlabel, fontsize=fontSize)
        ax.set_xlabel(plotXlabel, fontsize=fontSize)

        # more custom legend
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2, borderaxespad=0.)  # legend on top

    plt.show()



    # plt.ylabel('Accuracy (%)')
    # plt.xlabel('Epoch number')
    #plt.setp(lines, linewidth=2.0)

    # use keyword args
    # # or MATLAB style string value pairs
    # plt.setp(lines, 'color', 'r', 'linewidth', 2.0)

def getPerfData(networkList, audio_dataset='combined', runType=None):

    networkTrainInfos = []  #list of the TrainInfo dictionaries
    networkNames = []
    for network in networkList:
        if runType == None or network.runType == runType:
            networkPath = getNetworkTrainInfoPath(network, audio_dataset)
            networkName = fixNetworknameShort(network, networkPath)
            # networkTrainInfos.append(fixTrainInfo(network, unpickle(networkPath)))
            networkTrainInfos.append(unpickle(networkPath))

            networkNames.append(networkName)

    y = []
    for trainInfo in networkTrainInfos:
        try:
            y.append(max(trainInfo['test_acc']))
        except:
            try: y.append(trainInfo[audio_dataset+'final_test_acc'])
            except: y.append(trainInfo[audio_dataset+'test_acc'])
    x = range(1,len(y)+1)

    # sort networks based on accuracy
    #y, networkNames = (list(t) for t in zip(*sorted(zip(y, networkNames))))

    return x,y,networkNames, runType

def plotPerfData(x, y, networkNames, runType=None, numX =None):
    fontSizeNetworkNames = 20
    plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim(0, 100)
    if numX == None: ax.set_xlim(0.7, len(y)+0.3)
    else: ax.set_xlim(0.7, numX+1+0.3)
    ax.plot(x, y, linewidth = lineWidth)
    for i, j in zip(x, y):
        ax.annotate("{:10.3f}".format(j), xy=(i-0.05*(max(x)-min(x)), j+0.05*(max(y)-min(y))))


    if numX == None: pylab.xticks(x, networkNames, fontsize=fontSizeNetworkNames)
    else:  pylab.xticks(x[0:numX], networkNames[0:numX], fontsize=fontSizeNetworkNames)

    if runType == None: runType = 'all'
    ax.set_title('Performance for '+runType.capitalize()+' networks')
    ax.set_ylabel('Top 1 accuracy on Test Set', fontsize=fontSize)
    ax.set_xlabel('Network Architecture', fontsize=fontSize)
    plt.show()

    #
    # pylab.plot(x, counts, "g")
    #
    # pylab.show()

def plotSeries(seriesList, colors = None, labels=None, showNumbers=True):
    numX = len(seriesList[0][0])
    fontSizeNetworkNames = 16
    plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim(50, 100)

    ax.set_xlim(0.7,  numX + 0.3)
    x, y, networkNames, runType = seriesList[0]
    pylab.xticks(x, networkNames, fontsize=fontSizeNetworkNames)

    for i in range(len(seriesList)):
        series = seriesList[i]
        x, y, networkNames, _ = series
        if labels != None and colors != None:
            ax.plot(x, y, color=colors[i], label=labels[i],linewidth=lineWidth, linestyle='-', marker='o')
        elif colors!=None: ax.plot(x, y, color=colors[i], linewidth=lineWidth, linestyle='-', marker='o')
        else:            ax.plot(x, y, linewidth=lineWidth, linestyle='-', marker='o')
        if showNumbers:
            for i, j in zip(x, y):
                ax.annotate("{:10.3f}".format(j), xy=(i - 0.05 * (max(x) - min(x)), j + 0.05 * (max(y) - min(y))))


    if runType == None: runType = 'all'
    ax.set_title('Performance for ' + runType.capitalize() + ' networks')
    ax.set_ylabel('Top 1 accuracy on Test Set', fontsize=fontSize)
    ax.set_xlabel('Network Architecture', fontsize=fontSize)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5, fontsize=fontSize)

    # display = (0, 1, 2)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend([handle for i, handle in enumerate(handles) if i in display],
    #           [label for i, label in enumerate(labels) if i in display])
    #plt.show()

    fileName = "layersComparison"
    path = storeDir+fileName
    pylab.savefig(path+".png", bbox_inches='tight')
    #generateTexPlot(path, r'1\textwidth', dpi=300)

# audio accuracy are in [0,1] instaed of [0,100]
def fixTrainInfo(network, networkTrainInfo):
    print(networkTrainInfo['test_acc'])
    import re
    if network.runType == 'audio':
        for key in networkTrainInfo.keys():
            if re.match(r'(.*)acc', key):
                if type(networkTrainInfo[key]) == list:
                    networkTrainInfo[key] = [100.0 * val for val in networkTrainInfo[key]]
                else:
                    networkTrainInfo[key] = 100 * networkTrainInfo[key]
    #print(networkTrainInfo['test_acc'])
    return networkTrainInfo

def fixNetworkname(network, networkPath):
    import re
    # line = re.sub(r"""
    #   (?x) # Use free-spacing mode.
    #   <    # Match a literal '<'
    #   /?   # Optionally match a '/'
    #   \[   # Match a literal '['
    #   \d+  # Match one or more digits
    #   >    # Match a literal '>'
    #   """, "", line)
    networkName = os.path.basename(networkPath).replace(".npz","")
    networkName = re.sub(r"\d+\_LSTMLayer", "LSTM_", networkName)
    networkName = networkName.replace("_trainInfo.pkl", "") \
        .replace(network.datasetType, "") \
        .replace("_google_phoneme39_LSTM", "DeepmindCNN__LSTM") \
        .replace("_google_phoneme39", "DeepmindCNN") \
        .replace("__LSTM", "\nLSTM") \
        .replace("__CNN", "\nCNN") \
        .replace("__FC", "\nFC") \
        .replace("_nbMFCC39","") \
        .replace("_combined","") \
        .replace("__TCDTIMIT","")
    # .replace("bidirectional", "2-dir") \

    # add the network architecture in front
    if network.runType == 'lipreading':
        if network.LIP_RNN_HIDDEN_LIST == None:      networkName = network.runType + "_CNN\n" + networkName
        else:           networkName = network.runType + "_CNN_LSTM\n" + networkName
    else: networkName = network.runType + "\n" + networkName

    return networkName


def fixNetworknameShort(network, networkPath, type='audio'):
    import re
    # line = re.sub(r"""
    #   (?x) # Use free-spacing mode.
    #   <    # Match a literal '<'
    #   /?   # Optionally match a '/'
    #   \[   # Match a literal '['
    #   \d+  # Match one or more digits
    #   >    # Match a literal '>'
    #   """, "", line)
    networkName = os.path.basename(networkPath).replace(".npz", "")
    networkName = re.sub(r"\d+\_LSTMLayer", "LSTM_", networkName)
    networkName = networkName.replace("_trainInfo.pkl", "") \
        .replace(network.datasetType, "") \
        .replace("_google_phoneme39_LSTM", "DeepmindCNN__LSTM") \
        .replace("_google_phoneme39", "DeepmindCNN") \
        .replace("__LSTM", "\nLSTM") \
        .replace("__CNN", "\nCNN") \
        .replace("__FC", "\nFC") \
        .replace("_nbMFCC39", "") \
        .replace("_combined", "") \
        .replace("__TCDTIMIT", "")
    # .replace("bidirectional", "2-dir") \

    if type == 'audio':
        networkName = networkName.replace("_TIMIT","") \
                .replace("_TCDTIMIT", "") \
                .replace("_combined", "")\
                .replace("_bidirectional","") \
                .replace("LSTM_", "")

    return networkName


def plotWeightValues(network):

    # TODO: this is just the plot histogram example
    import numpy as np
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()


import matplotlib
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat
from pylatex.utils import italic
import os
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt  # noqa

def generateTexTable(fname, colTitles, rowNames, rowDataList):

    # make sure all rows have same length
    max_data_length = max([len(rowdata) for rowdata in rowDataList]); print("max data length: ", max_data_length)
    for rowdata in rowDataList:
        while len(rowdata) < max_data_length:
            rowdata += ['/']

    print colTitles
    colBaseTitle = colTitles[-1]
    colTitles = [colTitles[0]]
    j=0
    while len(colTitles) < max_data_length+1:
        colTitles.append(str(2**j) + " Layers " + colBaseTitle)
        j+=1


    geometry_options = {"tmargin": "1cm"}#, "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)

    with doc.create(Subsection('Table of something')):
        layout = ['|l']*len(colTitles)+['|']; layout = ''.join(layout);
        print(layout)
        with doc.create(Tabular(layout)) as table:
            table.add_hline()
            table.add_row((colTitles))
            table.add_hline()
            for rowName, rowData in zip(rowNames,rowDataList):
                rowList = [rowName] + rowData
                print rowList
                table.add_row(tuple(rowList))
                table.add_hline(1,1)
            table.add_hline()
    doc.generate_pdf(storeDir + fname, clean_tex=False)

if __name__ == '__main__':
    main()


