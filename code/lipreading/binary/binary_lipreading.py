from __future__ import print_function

import numpy as np

np.random.seed(1234)  # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T


import binary_net
import lasagne

from pylearn2.datasets.cifar10 import CIFAR10

from collections import OrderedDict

def main():
    # BN parameters
    batch_size = 80
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

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
    num_epochs = 20
    print("num_epochs = " + str(num_epochs))

    # Decaying LR 
    LR_start = 1e-5
    print("LR_start = " + str(LR_start))
    LR_fin = 1e-8
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))


    networkType = 'google'
    dataType = 'TCDTIMIT'

    model_name = os.path.expanduser('~/TCDTIMIT/lipreading/TCDTIMIT/results/CNN_binaryNet/lipspeakers_') \
                 + networkType + "_phoneme39_binary"
    model_path = model_name + ".npz"
    if not os.path.exists(os.path.dirname(model_path)): os.makedirs(os.path.dirname(model_path))


    print('Building the CNN...')
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = buildCNN(networkType, dataType, input, epsilon, alpha, activation, binary, stochastic, H, W_LR_scale)


    # restore network weights
    if os.path.exists(model_path):
        with np.load(model_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            try:
                lasagne.layers.set_all_param_values(cnn, *param_values)
            except:
                lasagne.layers.set_all_param_values(cnn, param_values)
            print("\t Loaded model " + model_path)


    print('Loading ' + dataType + ' dataset...')
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(dataType)


    print("Building Functions...")
    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    # W updates
    W = lasagne.layers.get_all_params(cnn, binary=True)
    W_grads = binary_net.compute_grads(loss, cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates, cnn)

    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
    updates = OrderedDict(
            updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')

    binary_net.train(
            train_fn, val_fn,
            cnn,
            batch_size,
            LR_start, LR_decay,
            num_epochs,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            save_name=model_name,
            shuffle_parts=shuffle_parts)

def buildCNN(networkType, dataType, input, epsilon, alpha, activation, binary, stochastic, H, W_LR_scale):
    if dataType =='TCDTIMIT':
        nbClasses = 39
        cnn = lasagne.layers.InputLayer(
                shape=(None, 1, 120,120),
                input_var=input)
    else:
        nbClasses = 10
        cnn = lasagne.layers.InputLayer(
                shape=(None, 3, 32, 32),
                input_var=input)

    if networkType == 'google':
        # conv 1
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
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
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
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
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        # conv 4
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        # conv 5
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        # FC layer
        cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,  #TODO was identity
                num_units=nbClasses)

        # cnn = lasagne.layers.BatchNormLayer(
        #         cnn,
        #         epsilon=epsilon,
        #         alpha=alpha)
    else:  # cifar10
        # 128C3-128C3-P2
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=128,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
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

        # 256C3-256C3-P2
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=256,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=256,
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

        # 512C3-512C3-P2
        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        cnn = binary_net.Conv2DLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512,
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

        # print(cnn.output_shape)

        # 1024FP-1024FP-10FP
        cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation)

        cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=nbClasses)

        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon,
                alpha=alpha)
    return cnn

import os
import preprocessLipreading
import general_tools
def loadDataset(type):
    if type == 'TCDTIMIT':
        nbClasses = 39
        oneHot = False
        # get the database
        # If it's small (lipspeakers) -> generate X_train, y_train etc here
        # otherwise we need to load and generate each speaker seperately in the training loop
        dataset = "TCDTIMIT"
        root_dir = os.path.join(os.path.expanduser('~/TCDTIMIT/lipreading/' + dataset))
        results_dir = root_dir + "/results/CNN_binaryNet";
        if not os.path.exists(results_dir): os.makedirs(results_dir)

        database_binaryDir = root_dir + '/binary'
        datasetType = "lipspeakers"  # "lipspeakers" #"volunteers" #"volunteers" #    lipspeakers or volunteers"
        ##############################################

        if datasetType == "lipspeakers":
            loadPerSpeaker = False  # only lipspeakers small enough to fit in CPU RAM, generate X_train etc here
            storeProcessed = True
            processedDir = database_binaryDir + "_allLipspeakersProcessed"

            # TODO: prepLip_all can be used to generate pkl containing all the lipspeaker data. Not sure if this stil works, so use with care!
            pkl_path = processedDir + os.sep + datasetType + "_oneHot" + ".pkl"
            if not os.path.exists(pkl_path):
                print("dataset not yet processed. Processing...")
                preprocessLipreading.prepLip_all(data_path=database_binaryDir, store_path=pkl_path, trainFraction=0.7,
                                                 validFraction=0.1,
                                                 testFraction=0.2,
                                                 nbClasses=nbClasses, onehot=oneHot, type=datasetType, verbose=True)
            X_train, y_train, X_val, y_val, X_test, y_test = general_tools.unpickle(pkl_path)
            dtypeX = 'float32'
            dtypeY = 'float32'
            X_train = X_train.astype(dtypeX);
            y_train = y_train.astype(dtypeY);
            X_val = X_val.astype(dtypeX);
            y_val = y_val.astype(dtypeY);
            X_test = X_test.astype(dtypeX);
            y_test = y_test.astype(dtypeY);

    else: # cifar10
        nbClasses = 10
        train_set_size = 45000
        print("train_set_size = " + str(train_set_size))
        train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
        valid_set = CIFAR10(which_set="train", start=train_set_size, stop=50000)
        test_set = CIFAR10(which_set="test")

        # bc01 format
        # Inputs in the range [-1,+1]
        # print("Inputs in the range [-1,+1]")
        train_set.X = np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 3, 32, 32))
        valid_set.X = np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 3, 32, 32))
        test_set.X = np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 3, 32, 32))

        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.

        X_train = train_set.X;
        y_train = train_set.y
        X_val = valid_set.X;
        y_val = valid_set.y
        X_test = test_set.X;
        y_test = test_set.y

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    main()
