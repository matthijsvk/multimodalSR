from __future__ import print_function

import os

import numpy as np

np.random.seed(1234)  # for reproducibility?

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import lasagne

os.environ["THEANO_FLAGS"] = "cuda.root=/usr/local/cuda,device=gpu,floatX=float32"
# specifying the gpu to use
import theano.sandbox.cuda
from collections import OrderedDict

theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

# from http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np

import logging
import code.lipreading.formatting

logger_lip = logging.getLogger('lipreading')
logger_lip.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(code.lipreading.formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger_lip.addHandler(ch)

# User - created files
import binary_net

import lasagne.layers as L

debug = True
binary = True

def main():
    # BN parameters
    batch_size = 100
    logger_lip.info("batch_size = %s", batch_size)
    # alpha is the exponential moving average factor
    alpha = .1
    logger_lip.info("alpha = %s", alpha)
    epsilon = 1e-4
    logger_lip.info("epsilon = %s", epsilon)

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_tanh_unit")
    stochastic = True
    print("stochastic = " + str(stochastic))
    # (-H,+H) are the two binary values
    #H = "Glorot"
    H = 1.
    print("H = " + str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"  # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = " + str(W_LR_scale))

    # Training parameters
    num_epochs = 50
    logger_lip.info("num_epochs = %s", num_epochs)

    # Decaying LR
    LR_start = 0.1
    logger_lip.info("LR_start = %s", LR_start)
    LR_fin = 0.0000003
    logger_lip.info("LR_fin = %s", LR_fin)
    # LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    LR_decay = 0.5  # sqrt(0.5)
    logger_lip.info("LR_decay = %s", LR_decay)
    # BTW, LR decay might good for the BN moving average...

    shuffle_parts = 1
    logger_lip.info("shuffle_parts = %s", shuffle_parts)
    if binary: oneHot = True
    else: oneHot = False

    ##############################################
    network_type = "google"
    viseme = False  # will set nbClasses and store path   vis: 6.498.828   phn: 7.176.231

    if viseme:
        nbClasses = 12
    else:
        nbClasses = 39

    justTest = False

    # get the database
    # If it's small (lipspeakers) -> generate X_train, y_train etc here
    # otherwise we need to load and generate each speaker seperately in the training loop
    dataset = "TCDTIMIT"
    root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset))
    results_dir = root_dir + "/results/CNN_binaryNet";
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if viseme:
        database_binaryDir = root_dir + '/binaryViseme'
    else:
        database_binaryDir = root_dir + '/binary'
    datasetType = "lipspeakers"  # "lipspeakers" #"volunteers" #"volunteers" #    lipspeakers or volunteers"
    ##############################################

    if datasetType == "lipspeakers":
        loadPerSpeaker = False  # only lipspeakers small enough to fit in CPU RAM, generate X_train etc here
        storeProcessed = True
        processedDir = database_binaryDir + "_allLipspeakersProcessed"

        # TODO: prepLip_all can be used to generate pkl containing all the lipspeaker data. Not sure if this stil works, so use with care!
        if not oneHot: pkl_path =  processedDir + os.sep + datasetType + ".pkl"
        else:
            pkl_path = processedDir + os.sep + datasetType + "_oneHot" + ".pkl"
        if not os.path.exists(pkl_path):
            logger_lip.info("dataset not yet processed. Processing...")
            code.lipreading.preprocessLipreading.prepLip_all(data_path=database_binaryDir, store_path=pkl_path, trainFraction=0.7, validFraction=0.1,
                                                             testFraction=0.2,
                                                             nbClasses=nbClasses, onehot=oneHot, type=datasetType, verbose=True)
        datasetFiles = code.lipreading.general_tools.unpickle(pkl_path)
        X_train, y_train, X_val, y_val, X_test, y_test = datasetFiles
        dtypeX = 'float32'
        dtypeY = 'float32'
        X_train = X_train.astype(dtypeX);
        y_train = y_train.astype(dtypeY);
        X_val = X_val.astype(dtypeX);
        y_val = y_val.astype(dtypeY);
        X_test = X_test.astype(dtypeX);
        y_test = y_test.astype(dtypeY);
        datasetFiles = [X_train, y_train, X_val, y_val, X_test, y_test]


        # These files have been generated with datasetToPkl_fromCombined, so that the train/val/test set are the same as for combinedSR.
        # X_train, y_train = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTrain.pkl"))
        # X_val, y_val = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersVal.pkl"))
        # X_test, y_test = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTest.pkl"))
        # datasetFiles = [X_train, y_train, X_val, y_val, X_test, y_test]

    else:  # we need to load and preprocess each speaker before we evaluate, because dataset is too large and doesn't fit in CPU RAM
        loadPerSpeaker = True
        storeProcessed = True  # if you have about 10GB hdd space, you can increase the speed by not reprocessing it each iteration
        processedDir = database_binaryDir + "_finalProcessed"
        # you can just run this program and it will generate the files the first time it encounters them, or generate them manually with datasetToPkl.py

        # just get the names
        testVolunteerNumbers = ["13F", "15F", "21M", "23M", "24M", "25M", "28M", "29M", "30F", "31F", "34M", "36F",
                                "37F", "43F", "47M", "51F", "54M"];
        testVolunteers = [str(testNumber) + ".pkl" for testNumber in testVolunteerNumbers];
        lipspeakers = ["Lipspkr1.pkl", "Lipspkr2.pkl", "Lipspkr3.pkl"];
        allSpeakers = [f for f in os.listdir(database_binaryDir) if
                       os.path.isfile(os.path.join(database_binaryDir, f)) and os.path.splitext(f)[1] == ".pkl"]
        trainVolunteers = [f for f in allSpeakers if not (f in testVolunteers or f in lipspeakers)];
        trainVolunteers = [vol for vol in trainVolunteers if vol is not None]

        if datasetType == "combined":
            trainingSpeakerFiles = trainVolunteers + lipspeakers
            testSpeakerFiles = testVolunteers
        elif datasetType == "volunteers":
            trainingSpeakerFiles = trainVolunteers
            testSpeakerFiles = testVolunteers
        else:
            raise Exception("invalid dataset entered")
        datasetFiles = [trainingSpeakerFiles, testSpeakerFiles]

    model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme") + str(nbClasses) \
        + ("_binary" if binary else "")
    model_save_name = os.path.join(results_dir, model_name)

    # log file
    logFile = results_dir + os.sep + model_name + '.log'
    # if os.path.exists(logFile):
    #     fh = logging.FileHandler(logFileT)  # append to existing log
    # else:
    fh = logging.FileHandler(logFile, 'w')  # create new logFile
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger_lip.addHandler(fh)

    logger_lip.info('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    inputs = T.tensor4('inputs')
    if oneHot:
        targets = T.matrix('targets')
    else:
        targets = T.ivector('targets')

    LR = T.scalar('LR', dtype=theano.config.floatX)

    # get the network structure
    l_out = code.lipreading.buildNetworks.build_network_google_binary(activation, alpha, epsilon, inputs, binary, stochastic, H,
                                                                      W_LR_scale)  # 7176231 params


    # print het amount of network parameters
    logger_lip.info("Using the %s network", network_type)
    logger_lip.info("The number of parameters of this network: %s", L.count_params(l_out))

    logger_lip.info("loading %s", model_save_name + '.npz')
    load_model(model_save_name + '.npz', l_out)


    logger_lip.info("* COMPILING FUNCTIONS...")
    train_output = lasagne.layers.get_output(l_out, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - targets * train_output)))

    # W updates
    W = lasagne.layers.get_all_params(l_out, binary=True)
    W_grads = binary_net.compute_grads(loss, l_out)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates, l_out)

    # other parameters updates
    params = lasagne.layers.get_all_params(l_out, trainable=True, binary=False)
    updates = OrderedDict(
            updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(l_out, deterministic=True)
    out_fn = theano.function([inputs], test_output)

    test_loss = T.mean(T.sqr(T.maximum(0., 1. - targets * test_output)))
    test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), T.argmax(targets, axis=1)), dtype=theano.config.floatX)
    k=3
    test_top3_acc = T.zeros((1,))
    topk_acc_fn = theano.function([],test_top3_acc)
    val_fn = theano.function([inputs,targets],[test_loss, test_acc, test_top3_acc])

    if debug:
        nb = 3
        debugX = X_train[0:nb]
        debugY = y_train[0:nb]
        out = out_fn(debugX)
        val = val_fn(debugX, debugY)
        import pdb;pdb.set_trace()

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([inputs, targets, LR], loss, updates=updates)

    logger_lip.info('Training...')
    import code.lipreading.train_lipreading
    code.lipreading.train_lipreading.train(
            train_fn=train_fn, val_fn=val_fn, out_fn=out_fn, topk_acc_fn=topk_acc_fn, k=k,
            network_output_layer=l_out,
            batch_size=batch_size,
            LR_start=LR_start, LR_decay=LR_decay,
            num_epochs=num_epochs,
            dataset=datasetFiles,
            database_binaryDir=database_binaryDir,
            storeProcessed=storeProcessed,
            processedDir=processedDir,
            loadPerSpeaker=loadPerSpeaker, justTest=justTest,
            save_name=model_save_name,
            shuffleEnabled=True)


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    a = cPickle.load(fo)
    fo.close()
    return a


def load_model(model_path, network_output_layer, logger=logger_lip):
    try:
        logger.info("Loading stored model...")
        # restore network weights
        with np.load(model_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            try:
                lasagne.layers.set_all_param_values(network_output_layer, param_values)
            except:
                lasagne.layers.set_all_param_values(network_output_layer, *param_values)

        logger.info("Loading parameters successful.")
        return 0

    except IOError as e:
        logger.info("%s", os.strerror(e.errno))
        logger.info('Model: %s not found. No weights loaded', model_path)
        return -1


if __name__ == "__main__":
    main()
