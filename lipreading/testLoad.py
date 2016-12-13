from __future__ import print_function

import numpy as np

# profile memory consumption using https://pypi.python.org/pypi/memory_profiler

np.random.seed(1234)  # for reproducibility?

import time
import os
os.environ["THEANO_FLAGS"] = "cuda.root=/usr/local/cuda,device=gpu,floatX=float32"
import theano
import theano.tensor as T
import lasagne
import numpy as np
import sys
import gc

import theano.sandbox.cuda
from theano import function, config, shared, sandbox

import cPickle as pickle
from sys import getsizeof
from lasagne import layers
from lasagne.updates import nesterov_momentum

# from http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from urllib import urlretrieve
# from nolearn.lasagne import NeuralNet
# from nolearn.lasagne import visualize
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

import logging
from theano.compat.six.moves import xrange
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial
from pylearn2.utils import string_utils

_logger = logging.getLogger(__name__)

# import user-created files
import train_lipreadingTCDTIMIT  # load training functions
from datasetClass import CIFAR10  # load the binary dataset in proper format
from resnet50 import *  # needed if you want the resnet network

    
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# testing
@profile
def my_lipreading ():

    # BN parameters
    batch_size = 64
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # activation
    activation = T.nnet.relu
    print("activation = T.nnet.relu")

    # Training parameters
    num_epochs = 50
    print("num_epochs = " + str(num_epochs))

    # Decaying LR
    LR_start = 0.001
    print("LR_start = " + str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))

    print('Loading TCDTIMIT dataset...')
    trainFraction = 0.8
    validFraction = 0.1
    testFraction = 0.1
    dtype = 'uint8'
    # lipspeaker 1: 14530 -> 14500
    # lipspeaker 2: 13000
    # lipspeaker 3: 14104 -> 14000
    # total =  14500 + 13000 + 14000 = 41500
    ntotal = 43000  # estimate, for initialization
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # prepare data to load
    fnames = ['Lipspkr%i.pkl' % i for i in range(1, 2)]  # all 3 lipsteakers
    datasets = {}
    datapath = os.path.join(os.path.expanduser('~/TCDTIMIT/database_binary'))
    for name in fnames:
        fname = os.path.join(datapath, name)
        if not os.path.exists(fname):
            raise IOError(fname + " was not found.")
        datasets[name] = fname #cache.datasetCache.cache_file(fname)

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
    print("We have empty matrices, memory required: ", memTot / 1000000, " MB")
    print("------------------------- 1 --------------------")
    time.sleep(0)

    # now load train data
    trainLoaded = 0
    validLoaded = 0
    testLoaded = 0

    for i, fname in enumerate(fnames):
        _logger.info('loading file %s' % datasets[fname])
        data= unpickle(datasets[fname])
                
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

    print("------------------------- 2 --------------------")
    time.sleep(0)

    # remove unneeded rows
    xtrain = np.cast['float32'](xtrain[0:trainLoaded])
    xvalid = np.cast['float32'](xvalid[0:validLoaded])
    xtest = np.cast['float32'](xtest[0:testLoaded])
    ytrain = ytrain[0:trainLoaded]
    yvalid = yvalid[0:validLoaded]
    ytest = ytest[0:testLoaded]

    memTot = xtrain.nbytes + xvalid.nbytes + xtest.nbytes + ytrain.nbytes + yvalid.nbytes + ytest.nbytes
    print("Total memory size required: ", memTot / 1000000, " MB")

    print("------------------------- 3 --------------------")
    time.sleep(0)
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

    print("preprocess done...")
    time.sleep(0)
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    print(xtest)
    
    def f1 (x):
        f = function([], sandbox.cuda.basic_ops.gpu_from_host(x*2.0/255 -1))
        return f()
            
    def scaleOnGpu(matrix):
        nbRows = matrix.shape[0]
        print("we have to scale", nbRows)
        done = 0
        batchLength = 2000
        thisBatchLength = batchLength
        i = 0
        while done != 1:
            if i + thisBatchLength > nbRows:
                print("last batch", i)
                done = 1
                thisBatchLength = nbRows - i
                print(thisBatchLength)
            # do the scaling on GPU
            matrix[i :(i + thisBatchLength), :] = f1(
                shared(matrix[i:(i + thisBatchLength), :]))
            i += batchLength
        return matrix
        
    xtrain = scaleOnGpu(xtrain)
    xvalid = scaleOnGpu(xvalid)
    xtest = scaleOnGpu(xtest)
    
    print(xtest)
    # xtrain = np.reshape(np.subtract(np.multiply(2. / 255., xtrain), 1.), (-1, 1, 120, 120))
    # xvalid = np.reshape(np.subtract(np.multiply(2. / 255., xvalid), 1.), (-1, 1, 120, 120))
    # xtest = np.reshape(np.subtract(np.multiply(2. / 255., xtest), 1.), (-1, 1, 120, 120))

    
    print("X done...")
    time.sleep(0)
    # # flatten targets
    # ytrain = np.hstack(ytrain)
    # yvalid = np.hstack(yvalid)
    # ytest = np.hstack(ytest)
    # Onehot the targets
    ytrain = np.float32(np.eye(39)[ytrain])
    yvalid = np.float32(np.eye(39)[yvalid])
    ytest = np.float32(np.eye(39)[ytest])
    # for hinge loss
    ytrain = 2 * ytrain - 1.
    yvalid = 2 * yvalid - 1.
    ytest = 2 * ytest - 1.

    print("the number of training examples is: ", len(xtrain))
    print("the number of valid examples is: ", len(xvalid))
    print("the number of test examples is: ", len(xtest))

    print("------------------------- 4 --------------------")
    time.sleep(0)
    
    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    # get the network structure
    # input
    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 120, 120),  # 5,120,120 (5 = #frames)
            input_var=input)
    # conv 1
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # conv 2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            stride=(2, 2),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # conv3
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # conv 4
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # conv 5
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # FC layer
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=39)

    train_output = lasagne.layers.get_output(cnn, deterministic=True)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    # set all params to trainable
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
            xtrain, ytrain,
            xvalid, yvalid,
            xtest, ytest,
            save_path="./googleNetBestModel.pkl",
            shuffle_parts=shuffle_parts)
    
if __name__ == '__main__':
    my_lipreading()
    
