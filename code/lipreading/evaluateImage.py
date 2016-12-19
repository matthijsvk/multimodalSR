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
import theano
import lasagne
import os
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Getting top 5 classes of images")

add_arg = parser.add_argument

add_arg("-i", "--input_image", help="Input image")
add_arg("-m", "--model_file", help="Model pickle file")

args = parser.parse_args()

# this imported file contains build_model(), which constructs the network structure that you van fill using the pkl file
# to generate the pkl file, you need to run the main function in resnet50CaffeToLasagne_ImageNet,
#   which populates the network from caffe, gets the classes and the mean image, and stores those in a pkl file
from lipreadingTCDTIMIT import *

def load_model (model_npz_file):
    if not os.path.exists(model_npz_file): print(
    "This npz file does not exist! Please run 'lipreadingTCDTIMIT' first to generate it.")

    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # activation
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    cnn = build_network_resnet50(input)
    
    with np.load('./results/ResNet50/allLipspeakers/allLipspeakers.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(cnn['prob'], param_values)
    return cnn


def prep_image (fname):
    im = np.array(Image.open(fname), dtype=np.uint8).flatten()
    im = np.subtract(np.multiply(2. / 255., im), 1.)
    im = np.reshape(im, (-1, 1, 120, 120))
    
    return im.astype('float32')


def get_net_fun (npz_model):
    net = load_model(npz_model)
    
    input = T.tensor4('inputs')
    # get_class_prob = theano.function([input], lasagne.layers.get_output(cnn, deterministic=True))
    get_class_prob = theano.function([net['input'].input_var],
                                     lasagne.layers.get_output(net['prob'], deterministic=True))
    def print_top5 (im_path):
        im = prep_image(im_path)
        prob = get_class_prob(im)[0]
        print(prob)
        phonemeNumberMap = getPhonemeNumberMap()
        pred = []
        for i in range(0,len(prob)):
            p = prob[i]
            prob_phoneme = phonemeNumberMap[str(i+1)]
            pred.append([prob_phoneme, p])
            # print(p, " ", prob_phoneme)
        pred = sorted(pred, key=lambda t: t[1], reverse=True)
        for p in pred:
            print(p)
    
    return get_class_prob, print_top5

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
    get_prob, print_top5 = get_net_fun(args.model_file)  # expects npz model
    t0 = time.clock()
    print_top5(args.input_image)
    t1 = time.clock()
    print("Total time taken {:.4f}".format(t1 - t0))

# Usage example
#python preprocessImage.py -i testImages/w.jpg
#python evaluateImage.py -i testImages/w_mouth_gray_resized.jpg -m results/ResNet50/allLipspeakers/allLipspeakers.npz