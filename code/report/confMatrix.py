import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
matplotlib.rcParams.update({'font.size': 22})



from general_tools import *
from phoneme_set import *

import theano
from theano import tensor as T
x = T.vector('x')
classes = T.scalar('n_classes')
onehot = T.eq(x.dimshuffle(0, 'x'), T.arange(classes).dimshuffle('x', 0))
oneHot = theano.function([x, classes], onehot)

examples = T.scalar('n_examples')
y = T.matrix('y')
y_pred = T.matrix('y_pred')
confMat = T.dot(y.T, y_pred) / examples
confusionMatrix = theano.function(inputs=[y, y_pred, examples], outputs=confMat)


def confusion_matrix(targets, preds, n_class):
    assert len(targets) == len(preds)
    return confusionMatrix(oneHot(targets, n_class), oneHot(preds, n_class), len(targets))

def plotConfusionMatrix(confMatrix, title=None, viseme=False):

    phonemeNames = [classToPhoneme39[i] for i in range(0, 39)]
    if viseme: phonemeNames = [classToViseme[i] for i in range(0,12)]
    fig = figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(confMatrix, cmap='binary')  # , cmap=plt.cm.Blues, alpha=0.10)
    if title != None: ax.set_title(title)
    ax.set_xlabel('Predictions', labelpad=50)
    ax.set_ylabel('True Class', labelpad=100)
    bins = range(0, len(phonemeNames))
    binsy = range(0, len(phonemeNames), 2)
    ax.set_xticks([]) # no numbers on axis
    ax.set_yticks([])
    bin_centers = 0.5 * np.hstack((np.diff(bins), np.diff(bins)[-1])) + bins
    for i, phoneme, x in zip(range(len(phonemeNames)),phonemeNames, bin_centers):
        if i%2 == 0:
            # Label the phonemes on x axis and y axis
            ax.annotate(phoneme, xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -10), textcoords='offset points', va='top', ha='center')
            ax.annotate(phoneme, xy=(0, x), xytext=(-40, 0), textcoords='offset points')
        else:
            # Label the phonemes on x axis and y axis
            ax.annotate(phoneme, xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -10), textcoords='offset points', va='top', ha='center')
            ax.annotate(phoneme, xy=(0, x), xytext=(-70, 0), textcoords='offset points')  #TODO: set -70 to -40 for viseme as there's plenty room anyway

    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.show()

# test example
# y_true = [2, 0, 2, 2, 0, 1]
# y_out = [0, 0, 2, 2, 0, 2]
# confMatrix = confusion_matrix(y_true, y_out, 3)  #rows= true class. cols = predicted class
# plotConfusionMatrix(confMatrix)
# import pdb;pdb.set_trace()

# The actual confusion matrices

CNN_path = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN/'
CNN_LSTM_path = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN_LSTM/'
combined_CNN_LSTM_path = '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/results/CNN_LSTM/'
audio_path = '/home/matthijs/TCDTIMIT/audioSR/'

confMatrices = []
titles = []

### PHONEMES ###
# phoneme lip
viseme=False

#phoneme CNN
confMatrices.append(unpickle(CNN_path + 'lipspeakers_google_phoneme39_confusionMatrix.pkl'))
titles.append("Lipreading, phoneme CNN")
# phoneme CNN + LSTM
confMatrices.append( unpickle(CNN_LSTM_path + 'lipspeakers_google_phoneme39_LSTM_bidirectional_256_256_conv_confusionMatrix.pkl'))
titles.append("Lipreading, phoneme CNN + LSTM")
#phoneme combined audio+CNN+LSTM
confMatrices.append(unpickle(combined_CNN_LSTM_path + \
    'lipspeakers/RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional__CNN_google_dense_lipRNN_256_256_RNNfeaturesDense__FC_512_512_512__TCDTIMIT_lipspeakers_confusionMatrix.pkl'))
titles.append("Multimodal SR, phoneme CNN + LSTM, audio LSTM")

for confMatrix, title in zip(confMatrices, titles):
    plotConfusionMatrix(confMatrix, title, viseme=viseme)

#### VISEMES ###
# confMatrices = []
# titles = []
# viseme=True
# confMatrices.append(unpickle(CNN_path + 'lipspeakers_cifar10_viseme39_confusionMatrix.pkl'))
# titles.append("Lipreading, viseme CNN")
# confMatrices.append(unpickle(CNN_LSTM_path + 'lipspeakers_google_viseme39_LSTM_bidirectional_256_256_conv_viseme_confusionMatrix.pkl'))
# titles.append("Lipreading, viseme CNN + LSTM")
#
# for confMatrix, title in zip(confMatrices, titles):
#     plotConfusionMatrix(confMatrix, title, viseme=viseme)

# # Audio
confMatrices = []
titles = []
viseme = False

confMatrices.append(unpickle(audio_path + 'combined/results/2_LSTMLayer256_256_nbMFCC39_bidirectional_combined_confusionMatrix.pkl'))
titles.append("combined Audio SR, 256 / 2 LSTM")

confMatrices.append(
    unpickle(audio_path + 'TCDTIMIT/results/2_LSTMLayer256_256_nbMFCC39_bidirectional_TCDTIMIT_confusionMatrix.pkl'))
titles.append("TCDTIMIT Audio SR, 256 / 2 LSTM")

for confMatrix, title in zip(confMatrices, titles):
    plotConfusionMatrix(confMatrix, title, viseme=viseme)



