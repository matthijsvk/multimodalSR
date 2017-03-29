from __future__ import print_function

import sys

print(sys.path)

import os

import numpy as np

np.random.seed(1234)  # for reproducibility?

import lasagne
import lasagne.layers

os.environ["THEANO_FLAGS"] = "cuda.root=/usr/local/cuda,device=gpu,floatX=float32"
# specifying the gpu to use
import theano.sandbox.cuda

theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

# from http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np

# from nolearn.lasagne import NeuralNet
# from nolearn.lasagne import visualize
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

import logging
from pylearn2.datasets import cache
from collections import OrderedDict

_logger = logging.getLogger(__name__)

# User - created files
import train_lipreadingTCDTIMIT  # load training functions
import datasetClass  # load the binary dataset in proper format
import buildNetworks


# import binary_net

def main():
    # BN parameters
    batch_size = 50
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # BinaryOut
    activation = binary_tanh_unit
    print("activation = binary_tanh_unit")
    # activation = binary_sigmoid_unit
    # print("activation = binary_sigmoid_unit")

    # BinaryConnect
    binary = True
    print("binary = " + str(binary))
    stochastic = False
    print("stochastic = " + str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = " + str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"  # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = " + str(W_LR_scale))

    # Training parameters
    num_epochs = 500
    print("num_epochs = " + str(num_epochs))

    # Decaying LR
    LR_start = 0.002
    print("LR_start = " + str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))

    print('Loading TCDTIMIT dataset...')
    database_binary_location = os.path.join(os.path.expanduser('~/TCDTIMIT/database_binary'))
    train_set, valid_set, test_set = load_dataset(database_binary_location, 0.8, 0.1,
                                                  0.1)  # location, %train, %valid, %test

    print("the number of training examples is: ", len(train_set.X))
    print("the number of valid examples is: ", len(valid_set.X))
    print("the number of test examples is: ", len(test_set.X))

    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    # get the network structure
    # cnn = buildNetworks.build_network_cifar10_binary(activation, alpha, epsilon, input, binary, stochastic, H, W_LR_scale) # 7176231 params
    cnn = buildNetworks.build_network_google_binary(activation, alpha, epsilon, input, binary, stochastic, H,
                                                    W_LR_scale)  # 7176231 params

    # print het amount of network parameters
    print("The number of parameters of this network: ", lasagne.layers.count_params(cnn))

    # get output layer, for calculating loss etc
    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    if binary:
        # W updates
        W = lasagne.layers.get_all_params(cnn, binary=True)
        W_grads = compute_grads(loss, cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = clipping_scaling(updates, cnn)

        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
        updates = OrderedDict(
                updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')

    train_lipreadingTCDTIMIT.train(
            train_fn, val_fn,
            cnn,
            batch_size,
            LR_start, LR_decay,
            num_epochs,
            train_set.X, train_set.y,
            valid_set.X, valid_set.y,
            test_set.X, test_set.y,
            save_path="./TCDTIMITBestModel",
            shuffle_parts=shuffle_parts)


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_dataset(datapath=os.path.join(os.path.expanduser('~/TCDTIMIT/database_binary')), trainFraction=0.8,
                 validFraction=0.1, testFraction=0.1):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10

    # Lipspeaker 1:                  14627 phonemes,    14617 extacted and useable
    # Lipspeaker 2:  28363 - 14627 = 13736 phonemes     13707 extracted
    # Lipspeaker 3:  42535 - 28363 = 14172 phonemes     14153 extracted
    # total Lipspeakers:  14500 + 13000 + 14000 = 42477

    dtype = 'uint8'
    ntotal = 50000  # estimate, for initialization. takes some safty margin
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # prepare data to load
    fnamesLipspkrs = ['Lipspkr%i.pkl' % i for i in range(1, 4)]  # all 3 lipsteakers
    fnamesVolunteers = []  # ['Volunteer%i.pkl' % i for i in range(1,11)]  # 12 first volunteers
    fnames = fnamesLipspkrs + fnamesVolunteers
    datasets = {}
    for name in fnames:
        fname = os.path.join(datapath, name)
        if not os.path.exists(fname):
            raise IOError(fname + " was not found.")
        datasets[name] = cache.datasetCache.cache_file(fname)

    # load the images
    # first initialize the matrices
    lenx = ntotal
    xtrain = np.zeros((lenx, img_size), dtype=dtype)
    xvalid = np.zeros((lenx, img_size), dtype=dtype)
    xtest = np.zeros((lenx, img_size), dtype=dtype)

    ytrain = np.zeros((lenx, 1), dtype=dtype)
    yvalid = np.zeros((lenx, 1), dtype=dtype)
    ytest = np.zeros((lenx, 1), dtype=dtype)

    # memory issues: print size
    memTot = xtrain.nbytes + xvalid.nbytes + xtest.nbytes + ytrain.nbytes + yvalid.nbytes + ytest.nbytes
    # print("Empty matrices, memory required: ", memTot / 1000000, " MB")

    # now load train data
    trainLoaded = 0
    validLoaded = 0
    testLoaded = 0

    for i, fname in enumerate(fnames):
        print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded, " out of ", ntotal)
        print("nbTrainLoaded: ", trainLoaded)
        print("nbValidLoaded: ", validLoaded)
        print("nbTestLoaded: ", testLoaded)

        print('loading file %s' % datasets[fname])
        data = unpickle(datasets[fname])

        thisN = data['data'].shape[0]
        print("This dataset contains ", thisN, " images")

        thisTrain = int(trainFraction * thisN)
        thisValid = int(validFraction * thisN)
        thisTest = thisN - thisTrain - thisValid  # compensates for rounding
        print("now loading : nbTrain, nbValid, nbTest")
        print("              ", thisTrain, thisValid, thisTest)

        xtrain[trainLoaded:trainLoaded + thisTrain, :] = data['data'][0:thisTrain]
        xvalid[validLoaded:validLoaded + thisValid, :] = data['data'][thisTrain:thisTrain + thisValid]
        xtest[testLoaded:testLoaded + thisTest, :] = data['data'][thisTrain + thisValid:thisN]

        ytrain[trainLoaded:trainLoaded + thisTrain, 0] = data['labels'][0:thisTrain]
        yvalid[validLoaded:validLoaded + thisValid, 0] = data['labels'][thisTrain:thisTrain + thisValid]
        ytest[testLoaded:testLoaded + thisTest, 0] = data['labels'][thisTrain + thisValid:thisN]

        trainLoaded += thisTrain
        validLoaded += thisValid
        testLoaded += thisTest

        if (trainLoaded + validLoaded + testLoaded) >= ntotal:
            print("loaded too many?")
            break

    ntest = testLoaded
    nvalid = validLoaded
    ntrain = trainLoaded
    print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded, " out of ", ntotal)
    print("nbTrainLoaded: ", trainLoaded)
    print("nbValidLoaded: ", validLoaded)
    print("nbTestLoaded: ", testLoaded)

    # remove unneeded rows
    xtrain = xtrain[0:trainLoaded]
    xvalid = xvalid[0:validLoaded]
    xtest = xtest[0:testLoaded]
    ytrain = ytrain[0:trainLoaded]
    yvalid = yvalid[0:validLoaded]
    ytest = ytest[0:testLoaded]

    memTot = xtrain.nbytes + xvalid.nbytes + xtest.nbytes + ytrain.nbytes + yvalid.nbytes + ytest.nbytes
    # print("Total memory size required: ", memTot / 1000000, " MB")

    # process this data, remove all zero rows (http://stackoverflow.com/questions/18397805/how-do-i-delete-a-row-in-a-np-array-which-contains-a-zero)
    # cast to numpy array
    if isinstance(ytrain, list):
        ytrain = np.asarray(ytrain).astype(dtype)
    if isinstance(yvalid, list):
        yvalid = np.asarray(yvalid).astype(dtype)
    if isinstance(ytest, list):
        ytest = np.asarray(ytest).astype(dtype)

    # fix labels (labels start at 1, but the library expects them to start at 0)
    ytrain = ytrain - 1
    yvalid = yvalid - 1
    ytest = ytest - 1

    # now, make objects with these matrices
    train_set = datasetClass.CIFAR10(xtrain, ytrain, img_shape)
    valid_set = datasetClass.CIFAR10(xvalid, yvalid, img_shape)
    test_set = datasetClass.CIFAR10(xtest, ytest, img_shape)

    # Inputs in the range [-1,+1]
    # def f1 (x):
    #     f = function([], sandbox.cuda.basic_ops.gpu_from_host(x * 2.0 / 255 - 1))
    #     return f()
    #
    # def scaleOnGpu (matrix):
    #     nbRows = matrix.shape[0]
    #     done = 0
    #     batchLength = 100
    #     thisBatchLength = batchLength
    #     i = 0
    #     while done != 1:
    #         if i + thisBatchLength > nbRows:
    #             done = 1
    #             thisBatchLength = nbRows - i
    #         # do the scaling on GPU
    #         matrix[i:(i + thisBatchLength), :] = f1(
    #                 shared(matrix[i:(i + thisBatchLength), :]))
    #         i += batchLength
    #     return matrix
    #
    # train_set.X = scaleOnGpu(train_set.X )
    # valid_set.X = scaleOnGpu(valid_set.X )
    # test_set.X = scaleOnGpu(test_set.X)

    train_set.X = np.subtract(np.multiply(2. / 255., train_set.X), 1.)
    valid_set.X = np.subtract(np.multiply(2. / 255., valid_set.X), 1.)
    test_set.X = np.subtract(np.multiply(2. / 255., test_set.X), 1.)

    train_set.X = np.reshape(train_set.X, (-1, 1, 120, 120))
    valid_set.X = np.reshape(valid_set.X, (-1, 1, 120, 120))
    test_set.X = np.reshape(test_set.X, (-1, 1, 120, 120))

    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    train_set.y = np.float32(np.eye(39)[train_set.y])
    valid_set.y = np.float32(np.eye(39)[valid_set.y])
    test_set.y = np.float32(np.eye(39)[test_set.y])
    # for hinge loss
    train_set.y = 2 * train_set.y - 1.
    valid_set.y = 2 * valid_set.y - 1.
    test_set.y = 2 * test_set.y - 1.

    return train_set, valid_set, test_set


############# BINARY NET  #######################3

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise


# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,


round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)


def hard_sigmoid(x):
    return T.clip((x + 1.) / 2., 0, 1)


# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2. * round3(hard_sigmoid(x)) - 1.


def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))


# This function computes the gradient of the binary weights
def compute_grads(loss, network):
    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))

    return grads


# This functions clips the weights after the parameter update
def clipping_scaling(updates, network):
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)

    for layer in layers:

        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = " + str(layer.W_LR_scale))
            print("H = " + str(layer.H))
            updates[param] = param + layer.W_LR_scale * (updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H, layer.H)

    return updates


if __name__ == "__main__":
    main()
