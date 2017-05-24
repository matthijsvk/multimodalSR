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
import formatting

logger_lip = logging.getLogger('lipreading')
logger_lip.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger_lip.addHandler(ch)


# User - created files
import train_lipreading  # load training functions
import buildNetworks
import preprocessLipreading
import general_tools

import lasagne.layers as L
import lasagne.objectives as LO


batch_sizes = [32]
networks = ["cifar10"]

justTest = False
viseme = False


def main():

    for batch_size, network_type in zip(batch_sizes, networks):
        print(batch_size, network_type)
        # BN parameters
        # batch_size = 100
        logger_lip.info("batch_size = %s",batch_size)
        # alpha is the exponential moving average factor
        alpha = .1
        logger_lip.info("alpha = %s",alpha)
        epsilon = 1e-4
        logger_lip.info("epsilon = %s",epsilon)

        # activation
        activation = T.nnet.relu
        logger_lip.info("activation = T.nnet.relu")

        # Training parameters
        num_epochs = 20
        logger_lip.info("num_epochs = %s", num_epochs)

        # Decaying LR
        LR_start = 0.001
        logger_lip.info("LR_start = %s", LR_start)
        LR_fin = 0.0000003
        logger_lip.info("LR_fin = %s",LR_fin)
        #LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
        LR_decay = 0.5  # sqrt(0.5)
        logger_lip.info("LR_decay = %s",LR_decay)
        # BTW, LR decay might good for the BN moving average...

        shuffle_parts = 1
        logger_lip.info("shuffle_parts = %s",shuffle_parts)
        oneHot = False

        ##############################################
        # network_type = "google" #"resnet50"

        if viseme:   nbClasses = 12
        else:        nbClasses = 39



        # get the database
        # If it's small (lipspeakers) -> generate X_train, y_train etc here
        # otherwise we need to load and generate each speaker seperately in the training loop
        dataset = "TCDTIMIT"
        root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/'+dataset))
        results_dir = root_dir + "/results/CNN";
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        if viseme: database_binaryDir = root_dir + '/binaryViseme'
        else:      database_binaryDir = root_dir + '/binary'
        datasetType = "lipspeakers"  #"volunteers" #    lipspeakers or volunteers"
        ##############################################

        if datasetType == "lipspeakers":
            loadPerSpeaker = False  # only lipspeakers small enough to fit in CPU RAM, generate X_train etc here
            storeProcessed = True
            processedDir = database_binaryDir + "_allLipspeakersProcessed"
            # pkl_path =  processedDir + os.sep + datasetType + ".pkl"
            # if not os.path.exists(pkl_path):
            #     logger_lip.info("dataset not yet processed. Processing...")
            #     preprocessLipreading.prepLip_all(data_path=database_binaryDir, store_path=pkl_path, trainFraction=0.7, validFraction=0.1,
            #                 testFraction=0.2,
            #                 nbClasses=nbClasses, onehot=oneHot, type=datasetType, verbose=True)
            #datasetFiles = general_tools.unpickle(pkl_path)

            # if this doesn't succeed, you probably have to generate the files with datasetToPkl_fromCombined.py
            X_train, y_train = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTrain.pkl"))
            X_val, y_val = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersVal.pkl"))
            X_test, y_test = unpickle(os.path.expanduser("~/TCDTIMIT/lipreading/TCDTIMIT/binary/allLipspeakersTest.pkl"))
            datasetFiles = [X_train, y_train, X_val, y_val, X_test, y_test]

        else:  # we need to load and preprocess each speaker before we evaluate, because dataset is too large and doesn't fit in CPU RAM
            loadPerSpeaker = True
            storeProcessed = True #if you have about 10GB hdd space, you can increase the speed by not reprocessing it each iteration
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

        model_name = datasetType + "_" + network_type + "_" + ("viseme" if viseme else "phoneme")+str(nbClasses)
        model_save_name = os.path.join(results_dir,model_name)


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
        if oneHot: targets = T.matrix('targets')
        else: targets = T.ivector('targets')

        LR = T.scalar('LR', dtype=theano.config.floatX)

        # get the network structure
        if network_type == "google":
            cnnDict, l_out = buildNetworks.build_network_google(activation, alpha, epsilon, inputs, nbClasses)  # 7.176.231 params
        elif network_type == "cifar10":
            cnn, l_out = buildNetworks.build_network_cifar10(input=inputs, nbClasses=nbClasses, activation=activation, alpha=alpha, epsilon=epsilon)
        elif network_type == "cifar10_v2":
            cnn, l_out = buildNetworks.build_network_cifar10_v2(input=inputs,nbClasses=nbClasses)
        elif network_type == "resnet50":
            cnn, l_out = buildNetworks.build_network_resnet50(inputs, nbClasses)

        # print het amount of network parameters
        logger_lip.info("Using the %s network", network_type)
        logger_lip.info("The number of parameters of this network: %s", L.count_params(l_out))


        logger_lip.info("loading %s", model_save_name + '.npz')
        load_model(model_save_name +'.npz', l_out)

        # a = '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/thirty.npz'
        # logger_lip.info("loading %s", a)
        # load_model(a, l_out)


        logger_lip.info("* COMPILING FUNCTIONS...")
        # for validation: disable dropout etc layers -> deterministic
        test_network_output = L.get_output(l_out, deterministic=True)
        test_acc = T.mean(T.eq(T.argmax(test_network_output, axis=1), targets),
                          dtype=theano.config.floatX)  # T.zeros((1,))
        test_loss = LO.categorical_crossentropy(test_network_output, targets);
        test_loss = test_loss.mean()

        # Top k accuracy
        k = 3
        # topk_acc = T.mean( T.any(T.eq(T.argsort(test_network_output, axis=1)[:, -k:], targets.dimshuffle(0, 'x')), axis=1),
        #     dtype=theano.config.floatX)
        topk_acc = T.mean(lasagne.objectives.categorical_accuracy(test_network_output, targets.flatten(), top_k=k))

        topk_acc_fn = theano.function([inputs, targets], topk_acc)

        val_fn = theano.function([inputs, targets], [test_loss, test_acc, topk_acc])


        # For training, use nondeterministic output
        network_output = L.get_output(l_out, deterministic=False)
        out_fn = theano.function([inputs], network_output)
        # cross-entropy loss
        loss = LO.categorical_crossentropy(network_output, targets);
        loss = loss.mean()
        # # Also add weight decay to the cost function
        weight_decay = 1e-5
        weightsl2 = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2)
        loss += weight_decay * weightsl2

        # acc
        err = T.mean(T.eq(T.argmax(network_output, axis=1), targets), dtype=theano.config.floatX)

        # set all params to trainable
        params = L.get_all_params(l_out, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
        # and returning the corresponding training loss:
        train_fn = theano.function([inputs, targets, LR], loss, updates=updates)

        logger_lip.info('Training...')

        train_lipreading.train(
            train_fn=train_fn, val_fn=val_fn, out_fn=out_fn, topk_acc_fn = topk_acc_fn, k=k,
            network_output_layer=l_out,
            batch_size=batch_size,
            LR_start=LR_start, LR_decay=LR_decay,
            num_epochs=num_epochs,
            dataset=datasetFiles,
            database_binaryDir=database_binaryDir,
            storeProcessed=storeProcessed,
            processedDir=processedDir,
            loadPerSpeaker=loadPerSpeaker, justTest =justTest,
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
                print(len(param_values));
                for layer in lasagne.layers.get_all_layers(network_output_layer):
                    print(layer)
                #import pdb;                pdb.set_trace();
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
