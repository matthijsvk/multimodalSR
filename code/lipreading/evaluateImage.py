from __future__ import print_function

import argparse
import time
from PIL import Image

# this imported file contains build_model(), which constructs the network structure that you van fill using the pkl file
# to generate the pkl file, you need to run the main function in resnet50CaffeToLasagne_ImageNet,
#   which populates the network from caffe, gets the classes and the mean image, and stores those in a pkl file
from lipreadingTCDTIMIT import *
from phoneme_set import phoneme_set_39, classToPhoneme39
import buildNetworks

nbClassesPhonemes = 39
nbClassesVisemes = 13

# TODO: very weird issue when trying to use the viseme file: seems like the stored network is for 39 features ipv 13??? Also super weird outputs, it's surely not trained for phonemes
# TODO: some of the stored networks also were genereated with different network structure than is in build_networks.py (more or larger dense layers for example)...


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

        elif networkType == 'resnet50':  # ResNet50
            cnnDict, outputLayer = buildNetworks.build_network_resnet50(inputs, nbClassesPhonemes)
            modelParameterFile = './results/Phoneme_trained/ResNet50/allLipspeakers/allLipspeakers.npz'
        else:
            raise Exception('ERROR: given network type unknown.')

    elif phonemeViseme == 'viseme':  # use viseme-trained network, only trained for Google Network
        cnnDict, outputLayer = buildNetworks.build_network_google(activation, alpha, epsilon, inputs, 39)#nbClassesVisemes)
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


def prep_image(fname):
    im = np.array(Image.open(fname), dtype=np.uint8).flatten()
    im = np.subtract(np.multiply(2. / 255., im), 1.)
    im = np.reshape(im, (-1, 1, 120, 120))

    return im.astype('float32')


def get_net_fun(phonemeViseme, networkType, k=5):
    outputLayer, inputs = load_model(phonemeViseme, networkType)

    target = T.ivector('targets')

    all_predictions = lasagne.layers.get_output(outputLayer, deterministic=True)
    get_all_prob = theano.function([inputs], all_predictions)

    maxprob = T.argmax(all_predictions, axis=1)
    get_first_prediction = theano.function([inputs], maxprob)

    accuracy = T.mean(T.eq(maxprob, target), dtype=theano.config.floatX)
    get_accuracy = theano.function([inputs, target], accuracy)

    # Top k accuracy
    topk_accuracy = T.mean(T.any(T.eq(T.argsort(all_predictions, axis=1)[:, -k:], target.dimshuffle(0, 'x')), axis=1),
                           dtype=theano.config.floatX)
    get_topk_accuracy = theano.function([inputs, target], topk_accuracy)

    def print_topk(im_path, k):
        im = prep_image(im_path)
        prob = get_all_prob(im)#[0]
        print(prob)
        phonemeNumberMap = classToPhoneme39
        pred = []
        for i in range(0, len(prob)):
            p = prob[i]
            prob_phoneme = phonemeNumberMap[i]
            pred.append([prob_phoneme, p])
            # print(p, " ", prob_phoneme)
        pred = sorted(pred, key=lambda t: t[1], reverse=True)
        pred = pred[:k]
        for p in pred:
            print(p)

    return get_all_prob, get_first_prediction, print_topk, get_accuracy, get_topk_accuracy


def prob_to_class(prob):
    a = []
    for p in list(prob):
        a.append(classToPhoneme39[p])
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting top results for this image...")
    add_arg = parser.add_argument
    add_arg("input_image", help="Input image to be evaluated")
    add_arg("target_phoneme", help="Correct phoneme of input image")
    add_arg("-n", "--network_type", help="Type of network to be used", default='google')
    add_arg("-p", "--output", help="Network outputting phonemes (1) or visemes (0)", default='phoneme')
    # add_arg("-m", "--model-file", help="Model pickle file that contains trained network parameters")
    args = parser.parse_args()

    print("Compiling functions...")
    get_all_prob, get_top1_prob, print_topk, get_accuracy, get_topk_accuracy = get_net_fun(args.output,
                                                                                           args.network_type,
                                                                                           10)  # expects npz model
    t0 = time.clock()
    print_topk(args.input_image, 10)
    t1 = time.clock()
    print("Total time taken {:.4f}".format(t1 - t0))

    target_phoneme = args.target_phoneme
    classNumber = np.array([phoneme_set_39[target_phoneme]]).astype('int32')

    im = prep_image(args.input_image)
    print(prob_to_class(get_top1_prob(im)))

    print("Top 1 accuracy: ", get_accuracy(im, classNumber))
    print("Top 5 accuracy: ", get_topk_accuracy(im, classNumber))

    # Usage example
    # (to extract mouth, convert to grayscale etc: python preprocessImage.py -i testImages/w.jpg)
    # python evaluateImage2.py testImages/sa1_120_aa.jpg aa