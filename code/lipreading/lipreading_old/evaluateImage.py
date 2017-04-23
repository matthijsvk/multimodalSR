from __future__ import print_function

import urllib
import io
import skimage.transform
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 8, 6
import argparse
import time
import pickle
import numpy as np

import lasagne
import theano
import os
import numpy as np
from PIL import Image

import buildNetworks

nbClassesPhonemes = 39
nbClassesVisemes = 12


parser = argparse.ArgumentParser(description="Getting top results for this image...")
add_arg = parser.add_argument
add_arg("-i", "--input-image", help="Input image to be evaluated")
add_arg("-n", "--network-type", help="Type of network to be used", default=1)
add_arg("-p", "--phoneme-trained", help="Network outputting phonemes (1) or visemes (0)", default=0)
#add_arg("-m", "--model-file", help="Model pickle file that contains trained network parameters")
args = parser.parse_args()

# this imported file contains build_model(), which constructs the network structure that you van fill using the pkl file
# to generate the pkl file, you need to run the main function in resnet50CaffeToLasagne_ImageNet,
#   which populates the network from caffe, gets the classes and the mean image, and stores those in a pkl file
from lipreadingTCDTIMIT import *


#  build the model structure, fill in the stored parameters from a trained network with this structure
#  networkType:  1 = CIFAR10, 2 = GoogleNet, 3 = ResNet50
#  phonemeViseme:  1 = phoneme-trained, 0 = viseme-trained (meaning outputs are visemes)
def load_model (phonemeViseme, networkType):
    # network parameters
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # activation
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    inputs = T.tensor4('inputs')
    targets = T.matrix('targets')

    if phonemeViseme ==1: #use phoneme-trained network
        if networkType == 1:  # CIFAR10
            cnn = buildNetworks.build_network_cifar10(activation, alpha, epsilon, inputs, nbClassesPhonemes) # nbClassesPhonemes = 39 (global variable)
            with np.load('./results/Phoneme_trained/CIFAR10/allLipspeakers/allLipspeakers.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(cnn, param_values)

        elif networkType == 2: #GoogleNet
            cnn = buildNetworks.build_network_google(activation, alpha, epsilon, inputs, nbClassesPhonemes)
            with np.load('./results/Phoneme_trained/GoogleNet/allLipspeakers/allLipspeakers.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(cnn, param_values)

        elif networkType == 3: #ResNet50
            cnn = buildNetworks.build_network_resnet50(inputs, nbClassesPhonemes)
            with np.load('./results/Phoneme_trained/ResNet50/allLipspeakers/allLipspeakers.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(cnn['prob'], param_values)
        else:
            print('ERROR: given network type unknown.')

    else:  #use viseme-trained network
        cnn = buildNetworks.build_network_google(activation, alpha, epsilon, inputs, nbClassesVisemes)  # nbClassesVisemes = 13 (global variable)
        with np.load('./results/Viseme_trained/GoogleNet/allLipspeakers.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(cnn, param_values)

    return cnn

# scale to [0-2], then substract 1 to center around 0 (so now all values are in [-1,1] area)
# then reshape to make the image fit the network input size
def prep_image (fname):
    im = np.array(Image.open(fname), dtype=np.uint8).flatten()
    im = np.subtract(np.multiply(2. / 255., im), 1.)
    im = np.reshape(im, (-1, 1, 120, 120))

    return im.astype('float32')

# functions that evaluate the network
#  networkType:  1 = CIFAR10, 2 = GoogleNet, 3 = ResNet50
def get_net_fun (phonemeViseme, networkType, numberShown=5):
    print("Loading model...")
    net = load_model(phonemeViseme, networkType)

    inputs = T.tensor4('inputs')
    target = T.tensor4('targets')
    k = 5 #get top-5 accuracy

    print("Compiling Theano evaluation functions...")
    if (networkType == 3): #ResNets needs a different way of evaluating
        prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
        get_class_prob = theano.function([net['input'].input_var], prediction)

    else:
        prediction = lasagne.layers.get_output(net, deterministic=True)
        get_class_prob = theano.function([inputs, target], prediction)

    # top 1 accuracy
    print("Printing prediction...")
    print(prediction)
    print("Calulating accuracy...")
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target), dtype=theano.config.floatX)
    # Top k accuracy
    accuracy_k = T.mean(T.any(T.eq(T.argsort(prediction, axis=1)[:, -k:], target.dimshuffle(0, 'x')), axis=1),
                        dtype=theano.config.floatX)
    print("Compilation done.")

    def print_top5 (im_path):
        print("Preprocessing image...")
        im = prep_image(im_path)
        print("Image preprocessed.")

        print("Evaluating image...")
        prob = get_class_prob(im)[0]
        print(prob)
        phonemeNumberMap = getPhonemeNumberMap()
        pred = []

        if (numberShown > len(prob) or numberShown < 1): #sanity check
            numberShown = len(prob)

        for i in range(0,numberShown): #print network output probabilities
            p = prob[i]
            prob_phoneme = phonemeNumberMap[str(i+1)]
            pred.append([prob_phoneme, p])
        pred = sorted(pred, key=lambda t: t[1], reverse=True)
        for p in pred:
            print(p)

        print("All done.")

    return get_class_prob, print_top5, accuracy, accuracy_k

def getPhonemeNumberMap (phonemeMap="./phonemeLabelConversion.txt"):
    phonemeNumberMap = {}
    with open(phonemeMap) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                phonemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = phoneme
                phonemeNumberMap[str(parts[1])] = parts[0]
    return phonemeNumberMap

# Lets take five images and compare prediction of Lasagne with Caffe
def test_lasagne_ImageNet (classes, image_urls, mean_values, net):
        im = prep_image(url, mean_values)
        prob = np.array(lasagne.layers.get_output(net['prob'], im, deterministic=True).eval())[0]

        print('LProbabilities: ')
        print(prob)

        res = sorted(prob_phoneme, key=lambda t: t, reverse=True)[:]
        for p in res:
            print('  ', p)

        plt.figure()
        plt.imshow(rawim.astype('uint8'))
        plt.axis('off')
        plt.show()

        print('\n\n')

if __name__ == "__main__":
    print("Compiling functions...")
    get_prob, print_top5, accuracy, accuracy_k = get_net_fun(1, 3, 10)  # argument = phonemeViseme, networkType, npz model, numberResultsShown
    print("the network had ", accuracy, " top 1 accuracy")
    print("the network had ", accuracy_k, " top 5 accuracy")

    t0 = time.clock()
    print_top5(args.input_image)
    t1 = time.clock()
    print("Total time taken {:.4f}".format(t1 - t0))

# Usage example
#python preprocessImage.py -i testImages/w.jpg
#python evaluateImage.py -i testImages/w_mouth_gray_resized.jpg -m results/ResNet50/allLipspeakers/allLipspeakers.npz
