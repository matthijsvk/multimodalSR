from __future__ import print_function

import os, sys

import numpy as np

np.random.seed(1234)  # for reproducibility?

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import lasagne

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

import logging
from pylearn2.datasets import cache

_logger = logging.getLogger(__name__)

# User - created files
import train_lipreadingTCDTIMIT  # load training functions
import datasetClass  # load the binary dataset in proper format
import buildNetworks

import lasagne.layers as L
import lasagne.objectives as LO


def main():
    # BN parameters
    batch_size = 4
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
    num_epochs = 20
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
    nbClasses = 39
    oneHot = False
    # database in binary format (pkl files)
    database_binary_location = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/database_binary'))
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_datasetImages(datapath=database_binary_location, trainFraction=0.8, validFraction=0.1, testFraction=0.1,
                                                  nbClasses=nbClasses, onehot=oneHot, type="all", verbose=True)

    print("the number of training examples is: ", len(train_X))
    print("the number of valid examples is: ", len(valid_X))
    print("the number of test examples is: ", len(test_X))

    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    inputs = T.tensor4('inputs')
    if oneHot: targets = T.matrix('targets')
    else: targets = T.ivector('targets')

    LR = T.scalar('LR', dtype=theano.config.floatX)

    # get the network structure
    print("Using Google network")
    cnnDict, l_out = buildNetworks.build_network_google(activation, alpha, epsilon, inputs, nbClasses)  # 7.176.231 params

    # print("Using CIFAR10 network")
    # cnn. l_out = buildNetworks.build_network_cifar10(activation, alpha, epsilon, input, nbClasses) # 9.074.087 params,    # with 2x FC1024: 23.634.855

    # print("Using ResNet50 network")
    # cnn, l_out = buildNetworks.build_network_resnet50(input, nbClasses)

    # print het amount of network parameters
    print("The number of parameters of this network: ", L.count_params(l_out))


    print("* COMPILING FUNCTIONS...")

    # for validation: disable dropout etc layers -> deterministic
    test_network_output = L.get_output(l_out, inputs, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_network_output, axis=1), targets), dtype=theano.config.floatX)
    test_loss = LO.aggregate(LO.categorical_crossentropy(test_network_output, targets))
    val_fn = theano.function([inputs, targets], [test_loss, test_err])

    # For training, use nondeterministic output
    network_output = L.get_output(l_out, deterministic=False)
    # cross-entropy loss
    loss = T.mean(LO.categorical_crossentropy(network_output, targets))
    # error
    err = T.mean(T.neq(T.argmax(network_output, axis=1), targets), dtype=theano.config.floatX)

    # set all params to trainable
    params = L.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([inputs, targets, LR], loss, updates=updates)



    print('Training...')

    train_lipreadingTCDTIMIT.train(
            train_fn, val_fn,
            l_out,
            batch_size,
            LR_start, LR_decay,
            num_epochs,
            train_X, train_y,
            valid_X, valid_y,
            test_X, test_y,
            save_path="./TCDTIMITBestModel",
            shuffle_parts=shuffle_parts)


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_datasetImages(datapath=os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/database_binary')), trainFraction=0.8,
                 validFraction=0.1, testFraction=0.1, nbClasses=39, onehot=False, type="all", nbLip=1, nbVol=54,verbose=False):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10

    # Lipspeaker 1:                  14627 phonemes,    14617 extacted and useable
    # Lipspeaker 2:  28363 - 14627 = 13736 phonemes     13707 extracted
    # Lipspeaker 3:  42535 - 28363 = 14172 phonemes     14153 extracted
    # total Lipspeakers:  14500 + 13000 + 14000 = 42477

    dtype = 'uint8'
    memAvaliableMB = 6000; memAvaliable = memAvaliableMB * 1024
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # prepare data to load
    fnamesLipspkrs = ['Lipspkr%i.pkl' % i for i in range(1,nbLip+1)]  # all 3 lipsteakers
    fnamesVolunteers = ['Volunteer%i.pkl' % i for i in range(1, nbVol+1)]  # some volunteers
    if type=="lipspeakers": fnames = fnamesLipspkrs
    elif type=="volunteers": fnames = fnamesVolunteers
    elif type =="all": fnames = fnamesLipspkrs + fnamesVolunteers
    else: raise Exception("wrong type of dataset entered")

    datasets = {}
    for name in fnames:
        fname = os.path.join(datapath, name)
        if not os.path.exists(fname):
            raise IOError(fname + " was not found.")
        datasets[name] = cache.datasetCache.cache_file(fname)

    # load the images
    # first initialize the matrices
    train_X = []; train_y = []
    valid_X = []; valid_y = []
    test_X = []; test_y = []

    # now load train data
    trainLoaded = 0
    validLoaded = 0
    testLoaded = 0

    for i, fname in enumerate(fnames):

        if verbose:
            print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded)
            print("nbTrainLoaded: ", trainLoaded)
            print("nbValidLoaded: ", validLoaded)
            print("nbTestLoaded: ", testLoaded)

        print('loading file %s' % datasets[fname])
        data = unpickle(datasets[fname])
        thisN = data['data'].shape[0]
        thisTrain = int(trainFraction * thisN)
        thisValid = int(validFraction * thisN)
        thisTest = thisN - thisTrain - thisValid  # compensates for rounding\
        if verbose:
            print("This dataset contains ", thisN, " images")
            print("now loading : nbTrain, nbValid, nbTest")
            print("              ", thisTrain, thisValid, thisTest)

        train_X = train_X + list(data['data'][0:thisTrain])
        valid_X = valid_X + list(data['data'][thisTrain:thisTrain + thisValid])
        test_X = test_X + list(data['data'][thisTrain + thisValid:thisN])

        train_y = train_y + list(data['labels'][0:thisTrain])
        valid_y = valid_y + list(data['labels'][thisTrain:thisTrain + thisValid])
        test_y = test_y + list(data['labels'][thisTrain + thisValid:thisN])

        trainLoaded += thisTrain
        validLoaded += thisValid
        testLoaded += thisTest
        if verbose:
            print("nbTrainLoaded: ", trainLoaded)
            print("nbValidLoaded: ", validLoaded)
            print("nbTestLoaded: ", testLoaded)
            print("Total loaded till now: ", trainLoaded + validLoaded + testLoaded)

        # estimate as float32 = 4* memory as uint8
        memEstimate = 4*(sys.getsizeof(train_X) + sys.getsizeof(valid_X) + sys.getsizeof(test_X) + \
                      sys.getsizeof(train_y) + sys.getsizeof(valid_y) + sys.getsizeof(test_y))
        if verbose: print("memory estimate: ", memEstimate/1000.0, "MB")
        # if memEstimate > 0.6 * memAvaliable:
        #     print("loaded too many for memory, stopping loading...")
        #     break

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32'  #needed for
    if isinstance(train_X, list):       train_X = np.asarray(train_X).astype(dtypeX);
    if isinstance(train_y, list):       train_y = np.asarray(train_y).astype(dtypeY);
    if isinstance(valid_X, list):       valid_X = np.asarray(valid_X).astype(dtypeX);
    if isinstance(valid_y, list):       valid_y = np.asarray(valid_y).astype(dtypeY);
    if isinstance(test_X, list):        test_X = np.asarray(test_X).astype(dtypeX);
    if isinstance(test_y, list):        test_y = np.asarray(test_y).astype(dtypeY);

    if verbose:
        print("TRAIN: ", train_X.shape, train_X[0][0].dtype)
        print(train_y.shape, train_y[0].dtype)
        print("VALID: ", valid_X.shape)
        print(valid_y.shape)
        print("TEST: ", test_X.shape)
        print(test_y.shape)

    memTot = train_X.nbytes + valid_X.nbytes + test_X.nbytes + train_y.nbytes + valid_y.nbytes + test_y.nbytes
    print("Total memory size required as float32: ", memTot / 1000000, " MB")

    # fix labels (labels start at 1, but the library expects them to start at 0)
    train_y = train_y - 1
    valid_y = valid_y - 1
    test_y = test_y - 1

    # rescale to interval [-1,1], cast to float32 for GPU use
    train_X = np.multiply(2. / 255., train_X, dtype = 'float32')
    train_X = np.subtract(train_X, 1., dtype='float32');
    valid_X = np.multiply(2. / 255., valid_X, dtype='float32')
    valid_X = np.subtract(valid_X, 1., dtype='float32');
    test_X = np.multiply(2. / 255., test_X, dtype='float32')
    test_X = np.subtract(test_X, 1., dtype='float32');

    if verbose:
        print("Train: ", train_X.shape, train_X[0][0].dtype)
        print("Valid: ", valid_X.shape, valid_X[0][0].dtype)
        print("Test: ", test_X.shape, test_X[0][0].dtype)

    # reshape to get one image per row
    train_X = np.reshape(train_X, (-1, 1, 120, 120))
    valid_X = np.reshape(valid_X, (-1, 1, 120, 120))
    test_X = np.reshape(test_X, (-1, 1, 120, 120))

    # also flatten targets to get one target per row
    # train_y = np.hstack(train_y)
    # valid_y = np.hstack(valid_y)
    # test_y = np.hstack(test_y)

    # Onehot the targets
    if onehot:
        train_y = np.float32(np.eye(nbClasses)[train_y])
        valid_y = np.float32(np.eye(nbClasses)[valid_y])
        test_y = np.float32(np.eye(nbClasses)[test_y])

    # for hinge loss
    train_y = 2 * train_y - 1.
    valid_y = 2 * valid_y - 1.
    test_y = 2 * test_y - 1.

    # cast to correct datatype, just to be sure. Everything needs to be float32 for GPU processing
    dtypeX = 'float32'
    dtypeY = 'int32'
    train_X = train_X.astype(dtypeX);
    train_y = train_y.astype(dtypeY);
    valid_X = valid_X.astype(dtypeX);
    valid_y = valid_y.astype(dtypeY);
    test_X = test_X.astype(dtypeX);
    test_y = test_y.astype(dtypeY);
    if verbose:
        print("\n Final datatype: ")
        print("TRAIN: ", train_X.shape, train_X[0][0].dtype)
        print(train_y.shape, train_y[0].dtype)
        print("VALID: ", valid_X.shape)
        print(valid_y.shape)
        print("TEST: ", test_X.shape)
        print(test_y.shape)

    ### STORE DATA ###
    dataList = [train_X, train_y, valid_X, valid_y, test_X, test_y]
    general_tools.saveToPkl(target_path, dataList)

    # these can be used to evaluate new data, so you don't have to load the whole dataset just to normalize
    meanStd_path = os.path.dirname(outputDir) + os.sep + os.path.basename(dataRootDir) + "MeanStd.pkl"
    logger.info('Saving Mean and Std_val to %s', meanStd_path)
    dataList = [mean_val, std_val]
    general_tools.saveToPkl(meanStd_path, dataList)

    return train_X, train_y, valid_X, valid_y, test_X, test_y


if __name__ == "__main__":
    main()
