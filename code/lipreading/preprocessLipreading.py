from pylearn2.datasets import cache
import numpy as np
import sys,os
import cPickle
import general_tools

import logging
logger_prepLip = logging.getLogger('lipreading.prep')
logger_prepLip.setLevel(logging.DEBUG)


def unpickle(file):
    fo = open(file, 'rb')
    a = cPickle.load(fo)
    fo.close()
    return a


def prepLip_one(speakerFile=None, trainFraction=0.7, validFraction=0.1,
                       nbClasses=39, onehot=False, verbose=False,  store_path=None):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10

    dtype = 'uint8'
    memAvaliableMB = 6000;
    memAvaliable = memAvaliableMB * 1024
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # load the images
    # first initialize the matrices
    X_train = [];   y_train = []
    X_val   = [];   y_val = []
    X_test  = [];   y_test = []

    logger_prepLip.info('loading file %s', speakerFile)
    data = unpickle(speakerFile)
    thisN = data['data'].shape[0]
    thisTrain = int(trainFraction * thisN)
    thisValid = int(validFraction * thisN)
    thisTest = thisN - thisTrain - thisValid  # compensates for rounding
    if trainFraction + validFraction == 1.0:
        thisValid = thisN - thisTrain; thisTest = 0

    if verbose:
        logger_prepLip.info("This dataset contains %s images", thisN)
        logger_prepLip.info("now loading : nbTrain, nbValid, nbTest")
        logger_prepLip.info("\t\t\t %s %s %s", thisTrain, thisValid, thisTest)

    X_train = X_train + list(data['data'][0:thisTrain])
    X_val   = X_val   + list(data['data'][thisTrain:thisTrain + thisValid])
    X_test  = X_test  + list(data['data'][thisTrain + thisValid:thisN])

    y_train = y_train + list(data['labels'][0:thisTrain])
    y_val   = y_val   + list(data['labels'][thisTrain:thisTrain + thisValid])
    y_test  = y_test  + list(data['labels'][thisTrain + thisValid:thisN])

    if verbose:
        logger_prepLip.info("nbTrainLoaded: ", len(X_train))
        logger_prepLip.info("nbValidLoaded: ", len(X_val))
        logger_prepLip.info("nbTestLoaded: ",  len(X_test))
        logger_prepLip.info("Total loaded: ", len(X_train) + len(X_val) + len(X_test))

    # estimate as float32 = 4* memory as uint8
    memEstimate = 4 * (sys.getsizeof(X_train) + sys.getsizeof(X_val) + sys.getsizeof(X_test) + \
                       sys.getsizeof(y_train) + sys.getsizeof(y_val) + sys.getsizeof(y_test))
    if verbose: logger_prepLip.info("memory estimate: %s", memEstimate / 1000.0, "MB")
    # if memEstimate > 0.6 * memAvaliable:
    #     logger_prepLip.info("loaded too many for memory, stopping loading...")
    #     break

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32'  # needed for
    if isinstance(X_train, list):       X_train = np.asarray(X_train).astype(dtypeX);
    if isinstance(y_train, list):       y_train = np.asarray(y_train).astype(dtypeY);
    if isinstance(X_val, list):         X_val = np.asarray(X_val).astype(dtypeX);
    if isinstance(y_val, list):         y_val = np.asarray(y_val).astype(dtypeY);
    if isinstance(X_test, list):        X_test = np.asarray(X_test).astype(dtypeX);
    if isinstance(y_test, list):        y_test = np.asarray(y_test).astype(dtypeY);

    if verbose:
        logger_prepLip.info("TRAIN: ", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info(y_train.shape, y_train[0].dtype)
        logger_prepLip.info("VALID: ", X_val.shape)
        logger_prepLip.info(y_val.shape)
        logger_prepLip.info("TEST: ", X_test.shape)
        logger_prepLip.info(y_test.shape)

    memTot = X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes
    logger_prepLip.info("Total memory size required as float32: %s MB", memTot / 1000000)

    # fix labels (labels start at 1, but the library expects them to start at 0)
    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    # rescale to interval [-1,1], cast to float32 for GPU use
    X_train = np.multiply(2. / 255., X_train, dtype='float32')
    X_train = np.subtract(X_train, 1., dtype='float32');
    X_val = np.multiply(2. / 255., X_val, dtype='float32')
    X_val = np.subtract(X_val, 1., dtype='float32');
    X_test = np.multiply(2. / 255., X_test, dtype='float32')
    X_test = np.subtract(X_test, 1., dtype='float32');

    if verbose:
        logger_prepLip.info("Train: ", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info("Valid: ", X_val.shape, X_val[0][0].dtype)
        logger_prepLip.info("Test: ", X_test.shape, X_test[0][0].dtype)

    # reshape to get one image per row
    X_train = np.reshape(X_train, (-1, 1, 120, 120))
    X_val = np.reshape(X_val, (-1, 1, 120, 120))
    X_test = np.reshape(X_test, (-1, 1, 120, 120))

    # also flatten targets to get one target per row
    # y_train = np.hstack(y_train)
    # y_val = np.hstack(y_val)
    # y_test = np.hstack(y_test)

    # Onehot the targets
    if onehot:
        y_train = np.float32(np.eye(nbClasses)[y_train])
        y_val = np.float32(np.eye(nbClasses)[y_val])
        y_test = np.float32(np.eye(nbClasses)[y_test])

    # cast to correct datatype, just to be sure. Everything needs to be float32 for GPU processing
    dtypeX = 'float32'
    dtypeY = 'int32'
    X_train = X_train.astype(dtypeX);
    y_train = y_train.astype(dtypeY);
    X_val = X_val.astype(dtypeX);
    y_val = y_val.astype(dtypeY);
    X_test = X_test.astype(dtypeX);
    y_test = y_test.astype(dtypeY);

    if verbose:
        logger_prepLip.info("\n Final datatype: ")
        logger_prepLip.info("TRAIN: ", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info(y_train.shape, y_train[0].dtype)
        logger_prepLip.info("VALID: ", X_val.shape)
        logger_prepLip.info(y_val.shape)
        logger_prepLip.info("TEST: ", X_test.shape)
        logger_prepLip.info(y_test.shape)

    ### STORE DATA ###
    dataList = [X_train, y_train, X_val, y_val, X_test, y_test]
    if store_path != None: general_tools.saveToPkl(store_path, dataList)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepLip_all(data_path=os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/database_binary/')),
                store_path=os.path.join(
                        os.path.expanduser('~/TCDTIMIT/lipreading/database_binaryprocessed/dataset.pkl')),
                type="all", nbLip=3, nbVol=54, trainFraction=0.8, validFraction=0.1, testFraction=0.1,
                nbClasses=39, onehot=False, verbose=False):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10

    # Lipspeaker 1:                  14627 phonemes,    14617 extacted and useable
    # Lipspeaker 2:  28363 - 14627 = 13736 phonemes     13707 extracted
    # Lipspeaker 3:  42535 - 28363 = 14172 phonemes     14153 extracted
    # total Lipspeakers:  14500 + 13000 + 14000 = 42477

    dtype = 'uint8'
    memAvaliableMB = 6000;
    memAvaliable = memAvaliableMB * 1024
    img_shape = (1, 120, 120)
    img_size = np.prod(img_shape)

    # prepare data to load
    fnamesLipspkrs = ['Lipspkr%i.pkl' % i for i in range(1, nbLip + 1)]  # all 3 lipsteakers
    fnamesVolunteers = ['Volunteer%i.pkl' % i for i in range(1, nbVol + 1)]  # some volunteers
    if type == "lipspeakers":
        fnames = fnamesLipspkrs
    elif type == "volunteers":
        fnames = fnamesVolunteers
    elif type == "all":
        fnames = fnamesLipspkrs + fnamesVolunteers
    else:
        raise Exception("wrong type of dataset entered")

    datasets = {}
    for name in fnames:
        fname = os.path.join(data_path, name)
        if not os.path.exists(fname):
            raise IOError(fname + " was not found.")
        datasets[name] = cache.datasetCache.cache_file(fname)

    # load the images
    # first initialize the matrices
    X_train = [];
    y_train = []
    X_val = [];
    y_val = []
    X_test = [];
    y_test = []

    # now load train data
    trainLoaded = 0
    validLoaded = 0
    testLoaded = 0

    for i, fname in enumerate(fnames):

        if verbose:
            logger_prepLip.info("Total loaded till now: %s ", trainLoaded + validLoaded + testLoaded)
            logger_prepLip.info("nbTrainLoaded:  %s", trainLoaded)
            logger_prepLip.info("nbValidLoaded:  %s", validLoaded)
            logger_prepLip.info("nbTestLoaded:   %s", testLoaded)

        logger_prepLip.info('loading file %s' % datasets[fname])
        data = unpickle(datasets[fname])
        thisN = data['data'].shape[0]
        thisTrain = int(trainFraction * thisN)
        thisValid = int(validFraction * thisN)
        thisTest = thisN - thisTrain - thisValid  # compensates for rounding\
        if verbose:
            logger_prepLip.info("This dataset contains %s images", thisN)
            logger_prepLip.info("now loading : nbTrain, nbValid, nbTest")
            logger_prepLip.info("\t\t\t %s %s %s ", thisTrain, thisValid, thisTest)

        X_train = X_train + list(data['data'][0:thisTrain])
        X_val = X_val + list(data['data'][thisTrain:thisTrain + thisValid])
        X_test = X_test + list(data['data'][thisTrain + thisValid:thisN])

        y_train = y_train + list(data['labels'][0:thisTrain])
        y_val = y_val + list(data['labels'][thisTrain:thisTrain + thisValid])
        y_test = y_test + list(data['labels'][thisTrain + thisValid:thisN])

        trainLoaded += thisTrain
        validLoaded += thisValid
        testLoaded += thisTest
        if verbose:
            logger_prepLip.info("nbTrainLoaded:  %s", trainLoaded)
            logger_prepLip.info("nbValidLoaded:  %s", validLoaded)
            logger_prepLip.info("nbTestLoaded:   %s", testLoaded)
            logger_prepLip.info("Total loaded till now: ", trainLoaded + validLoaded + testLoaded)

        # estimate as float32 = 4* memory as uint8
        memEstimate = 4 * (sys.getsizeof(X_train) + sys.getsizeof(X_val) + sys.getsizeof(X_test) + \
                           sys.getsizeof(y_train) + sys.getsizeof(y_val) + sys.getsizeof(y_test))
        if verbose: logger_prepLip.info("memory estimate: %s", memEstimate / 1000.0, "MB")
        # if memEstimate > 0.6 * memAvaliable:
        #     logger_prepLip.info("loaded too many for memory, stopping loading...")
        #     break

    # cast to numpy array, correct datatype
    dtypeX = 'float32'
    dtypeY = 'int32'  # needed for
    if isinstance(X_train, list):       X_train = np.asarray(X_train).astype(dtypeX);
    if isinstance(y_train, list):       y_train = np.asarray(y_train).astype(dtypeY);
    if isinstance(X_val, list):       X_val = np.asarray(X_val).astype(dtypeX);
    if isinstance(y_val, list):       y_val = np.asarray(y_val).astype(dtypeY);
    if isinstance(X_test, list):        X_test = np.asarray(X_test).astype(dtypeX);
    if isinstance(y_test, list):        y_test = np.asarray(y_test).astype(dtypeY);

    if verbose:
        logger_prepLip.info("TRAIN: %s %s", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info(y_train.shape, y_train[0].dtype)
        logger_prepLip.info("VALID: %s", X_val.shape)
        logger_prepLip.info(y_val.shape)
        logger_prepLip.info("TEST: %s", X_test.shape)
        logger_prepLip.info(y_test.shape)

    memTot = X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes
    logger_prepLip.info("Total memory size required as float32: %s", memTot / 1000000, " MB")

    # fix labels (labels start at 1, but the library expects them to start at 0)
    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    # rescale to interval [-1,1], cast to float32 for GPU use
    X_train = np.multiply(2. / 255., X_train, dtype='float32')
    X_train = np.subtract(X_train, 1., dtype='float32');
    X_val = np.multiply(2. / 255., X_val, dtype='float32')
    X_val = np.subtract(X_val, 1., dtype='float32');
    X_test = np.multiply(2. / 255., X_test, dtype='float32')
    X_test = np.subtract(X_test, 1., dtype='float32');

    if verbose:
        logger_prepLip.info("Train: %s %s", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info("Valid: %s %s", X_val.shape, X_val[0][0].dtype)
        logger_prepLip.info("Test: %s %s", X_test.shape, X_test[0][0].dtype)

    # reshape to get one image per row
    X_train = np.reshape(X_train, (-1, 1, 120, 120))
    X_val = np.reshape(X_val, (-1, 1, 120, 120))
    X_test = np.reshape(X_test, (-1, 1, 120, 120))

    # also flatten targets to get one target per row
    # y_train = np.hstack(y_train)
    # y_val = np.hstack(y_val)
    # y_test = np.hstack(y_test)

    # Onehot the targets
    if onehot:
        y_train = np.float32(np.eye(nbClasses)[y_train])
        y_val = np.float32(np.eye(nbClasses)[y_val])
        y_test = np.float32(np.eye(nbClasses)[y_test])

    # for hinge loss
    # y_train = 2 * y_train - 1.
    # y_val = 2 * y_val - 1.
    # y_test = 2 * y_test - 1.

    # cast to correct datatype, just to be sure. Everything needs to be float32 for GPU processing
    dtypeX = 'float32'
    dtypeY = 'int32'
    X_train = X_train.astype(dtypeX);
    y_train = y_train.astype(dtypeY);
    X_val = X_val.astype(dtypeX);
    y_val = y_val.astype(dtypeY);
    X_test = X_test.astype(dtypeX);
    y_test = y_test.astype(dtypeY);
    if verbose:
        logger_prepLip.info("\n Final datatype: ")
        logger_prepLip.info("TRAIN: %s %s ", X_train.shape, X_train[0][0].dtype)
        logger_prepLip.info(y_train.shape, y_train[0].dtype)
        logger_prepLip.info("VALID: %s", X_val.shape)
        logger_prepLip.info(y_val.shape)
        logger_prepLip.info("TEST: %s", X_test.shape)
        logger_prepLip.info(y_test.shape)

    ### STORE DATA ###
    dataList = [X_train, y_train, X_val, y_val, X_test, y_test]
    general_tools.saveToPkl(store_path, dataList)

    return X_train, y_train, X_val, y_val, X_test, y_test