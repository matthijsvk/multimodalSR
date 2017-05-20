from __future__ import print_function

from lipreading import *
from phoneme_set import phoneme_set_39, classToPhoneme39
import argparse

nbClassesPhonemes = 39
nbClassesVisemes = 12

def main():
    parser = argparse.ArgumentParser(description="Getting top results for this image...")
    add_arg = parser.add_argument
    add_arg("-n", "--network_type", help="Type of network to be used", default='google')
    add_arg("-p", "--output", help="Network outputting phonemes (1) or visemes (0)", default='phoneme')
    # add_arg("-m", "--model-file", help="Model pickle file that contains trained network parameters")
    args = parser.parse_args()

    X_test, y_test = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTest.pkl"))
    evaluateNetwork(X_test, y_test, phonemeViseme="phoneme", networkType=args.network_type)


#  build the model structure, fill in the stored parameters from a trained network with this structure
#  networkType:  1 = CIFAR10, 2 = GoogleNet, 3 = ResNet50
#  phonemeViseme:  1 = phoneme-trained, 0 = viseme-trained (meaning outputs are visemes)
def load_model(phonemeViseme, networkType, printNetwork=False):
    # network parameters
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))
    activation = T.nnet.relu
    print("activation = T.nnet.relu")

    inputs = T.tensor4('inputs')

    if phonemeViseme == 'phoneme':  # use phoneme-trained network
        if networkType == 'cifar10':  # CIFAR10
            cnnDict, outputLayer = buildNetworks.build_network_cifar10(activation, alpha, epsilon, inputs,
                                                                       nbClassesPhonemes)
            modelParameterFile = './results/Phoneme_trained/CIFAR10/allLipspeakers/allLipspeakers.npz'

        elif networkType == 'google':  # GoogleNet
            cnnDict, outputLayer = buildNetworks.build_network_google(activation, alpha, epsilon, inputs,
                                                                      nbClassesPhonemes)
            modelParameterFile = './results/Phoneme_trained/GoogleNet/allLipspeakers/allLipspeakers.npz'
            modelParameterFile = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN/lipspeakers_google_phoneme39.npz'

        elif networkType == 'resnet50':  # ResNet50
            cnnDict, outputLayer = buildNetworks.build_network_resnet50(inputs, nbClassesPhonemes)
            modelParameterFile = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN/lipspeakers_resnet50_phoneme39.npz'

            modelParameterFile = './results/Phoneme_trained/ResNet50/allLipspeakers/allLipspeakers.npz'
        else:
            raise Exception('ERROR: given network type unknown.')

    elif phonemeViseme == 'viseme':  # use viseme-trained network, only trained for Google Network
        cnnDict, outputLayer = buildNetworks.build_network_google(activation, alpha, epsilon, inputs,
                                                                  nbClassesVisemes)
        modelParameterFile = './results/Viseme_trained/GoogleNet/allLipspeakers.npz'

    else:
        raise Exception('ERROR: given phoneme viseme type unknown.')

    # print the network structure
    if printNetwork: print_cnnNetwork(cnnDict)

    # load all the parameters
    with np.load(modelParameterFile) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(outputLayer, param_values)

    return outputLayer, inputs


def print_cnnNetwork(cnnDict):
    print("\n PRINTING Network structure: \n %s " % (sorted(cnnDict.keys())))
    for key in sorted(cnnDict.keys()):
        print(key)
        if 'conv' in key and type(cnnDict[key]) == list:
            for layer in cnnDict[key]:
                try:
                    print('      %12s \nin: %s | out: %s' % (layer, layer.input_shape, layer.output_shape))
                except:
                    print('      %12s \nout: %s' % (layer, layer.output_shape))
        else:
            try:
                print('Layer: %12s \nin: %s | out: %s' % (
                    cnnDict[key], cnnDict[key].input_shape, cnnDict[key].output_shape))
            except:
                print('Layer: %12s \nout: %s' % (cnnDict[key], cnnDict[key].output_shape))
    return 0


def getPhonemeToVisemeMap():
    map = {
        'f':   'A', 'v': 'A',
        'er':  'B', 'ow': 'B', 'r': 'B', 'q': 'B', 'w': 'B', 'uh': 'B', 'uw': 'B', 'axr': 'B', 'ux': 'B',
        'b':   'C', 'p': 'C', 'm': 'C', 'em': 'C',
        'aw':  'D',
        ' dh': 'E', 'th': 'E',
        'ch':  'F', 'jh': 'F', 'sh': 'F', 'zh': 'F',
        'oy':  'G', 'ao': 'G',
        's':   'H', 'z': 'H',
        'aa':  'I', 'ae': 'I', 'ah': 'I', 'ay': 'I', 'ey': 'I', 'ih': 'I', 'iy': 'I', 'y': 'I', 'eh': 'I', 'ax-h': 'I',
        'ax':  'I', 'ix': 'I',
        'd':   'J', 'l': 'J', 'n': 'J', 't': 'J', 'el': 'J', 'nx': 'J', 'en': 'J', 'dx': 'J',
        'g':   'K', 'k': 'K', 'ng': 'K', 'eng': 'K',
        'sil': 'S', 'pcl': 'S', 'tcl': 'S', 'kcl': 'S', 'bcl': 'S', 'dcl': 'S', 'gcl': 'S', 'h#': 'S', '#h': 'S',
        'pau': 'S', 'epi': 'S'
    }
    return map


def prob_to_class(prob):
    a = []
    for p in list(prob):
        a.append(classToPhoneme39[p])
    return a


from phoneme_set import phoneme_set_39, classToPhoneme39
def getPhonemeNumberMap():
    z = phoneme_set_39.copy()
    z.update(classToPhoneme39)
    return

batch_size = 32
def evaluateNetwork(X, y, phonemeViseme, networkType):
    phonemeToViseme = getPhonemeToVisemeMap()
    phonemeNumberMap = getPhonemeNumberMap()  # bidirectional map phoneme-number

    cnn = load_model(phonemeViseme, networkType)


    input = T.tensor4('inputs')
    target = T.matrix('targets')

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    val_fn = theano.function([input, target], [test_loss, test_err])

    # calculate validation error of whole dataset
    val_err = 0
    val_loss = 0
    batches = len(X) / batch_size

    for i in range(batches):
        new_loss, new_err = val_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
        val_err += new_err
        val_loss += new_loss

    val_err = val_err / batches * 100
    val_loss /= batches

    print("  validation loss:               " + str(val_loss))
    print("  validation error rate:         " + str(val_err) + "%")
    print("  test loss:                     " + str(test_loss))
    print("  test error rate:               " + str(test_err) + "%")


# ## Confusion matrix
# import numpy as np
#
# import theano
# from theano import tensor as T
#
# x = T.vector('x')
# classes = T.scalar('n_classes')
# onehot = T.eq(x.dimshuffle(0, 'x'), T.arange(classes).dimshuffle('x', 0))
# oneHot = theano.function([x, classes], onehot)
#
# examples = T.scalar('n_examples')
# y = T.matrix('y')
# y_pred = T.matrix('y_pred')
# confMat = T.dot(y.T, y_pred) / examples
# confusionMatrix = theano.function(inputs=[y, y_pred, examples], outputs=confMat)
#
#
# def confusion_matrix(targets, preds, n_class):
#     assert len(targets) == len(preds)
#     return confusionMatrix(oneHot(targets, n_class), oneHot(preds, n_class), len(targets))
#
#
# y_true = [2, 0, 2, 2, 0, 1]
# y_out = [0, 0, 2, 2, 0, 2]
# print(confusion_matrix(y_true, y_out, 3))  # rows= true class. cols = predicted class


if __name__ == "__main__":
    main()
