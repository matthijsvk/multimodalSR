import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True Class')
    bins = range(0, len(phonemeNames))
    ax.set_xticks(bins)
    ax.set_yticks(bins)
    bin_centers = 0.5 * np.hstack((np.diff(bins), np.diff(bins)[-1])) + bins
    for phoneme, x in zip(phonemeNames, bin_centers):
        # Label the phonemes on x axis and y axis
        ax.annotate(phoneme, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')
        ax.annotate(phoneme, xy=(0, x), xytext=(-60, 0), textcoords='offset points')

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

CNN_path = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN/'
CNN_LSTM_path = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN_LSTM/'
combined_CNN_LSTM_path = '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/results/CNN_LSTM/'

confMatrices = []
titles = []

### PHONEMES ###
# phoneme lip
viseme=False
confMatrices.append(unpickle(CNN_path + 'lipspeakers_google_phoneme39_confusionMatrix.pkl'))
titles.append("Lipreading, phoneme CNN")
# phoneme lip + LSTM
confMatrices.append( unpickle(CNN_LSTM_path + 'lipspeakers_google_phoneme39_LSTM_bidirectional_256_256_conv_confusionMatrix.pkl'))
titles.append("Lipreading, phoneme CNN + LSTM")
# phoneme combined
confMatrices.append(unpickle(combined_CNN_LSTM_path + \
    'lipspeakers/RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional__CNN_google_dense_lipRNN_256_256_RNNfeaturesDense__FC_512_512_512__TCDTIMIT_lipspeakers_confusionMatrix.pkl'))
titles.append("Multimodal SR, phoneme CNN + LSTM, audio LSTM")

for confMatrix, title in zip(confMatrices, titles):
    plotConfusionMatrix(confMatrix, title, viseme=viseme)

#### VISEMES ###
confMatrices = []
titles = []
viseme=True
confMatrices.append(unpickle(CNN_path + 'lipspeakers_cifar10_viseme39_confusionMatrix.pkl'))
titles.append("Lipreading, viseme CNN")
confMatrices.append(unpickle(CNN_LSTM_path + 'lipspeakers_google_viseme39_LSTM_bidirectional_256_256_conv_viseme_confusionMatrix.pkl'))
titles.append("Lipreading, viseme CNN + LSTM")

for confMatrix, title in zip(confMatrices, titles):
    plotConfusionMatrix(confMatrix, title, viseme=viseme)


