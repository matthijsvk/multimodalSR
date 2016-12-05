from __future__ import print_function

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import os
os.environ["THEANO_FLAGS"] = "cuda.root=/usr/local/cuda,device=gpu,floatX=float32"
import theano
import theano.tensor as T

import lasagne

import train_net

from pylearn2.datasets.cifar10 import CIFAR10

def main():
    # BN parameters
    batch_size = 50
    print( "batch_size = " +str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print( "alpha = " +str(alpha))
    epsilon = 1e-4
    print( "epsilon = " +str(epsilon))
    
    # activation
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    
    # Training parameters
    num_epochs = 20
    print( "num_epochs = " +str(num_epochs))
    
    # Decaying LR
    LR_start = 0.001
    print( "LR_start = " +str(LR_start))
    LR_fin = 0.0000003
    print( "LR_fin = " +str(LR_fin))
    LR_decay = ( LR_fin /LR_start ) **( 1. /num_epochs)
    print( "LR_decay = " +str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    train_set_size = 45000
    print( "train_set_size = " +str(train_set_size))
    shuffle_parts = 1
    print( "shuffle_parts = " +str(shuffle_parts))
    
    print('Loading CIFAR-10 dataset...')

    train_set, valid_set, test_set = load_dataset(train_set_size)
    
    print('Building the CNN...')
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    # get the network structure
    # cnn = build_network()

    cnn = build_network(activation, alpha, epsilon, input)
    
    # get output layer, for calculating loss etc
    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
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
    
    train_net.train(
            train_fn, val_fn,
            cnn,
            batch_size,
            LR_start, LR_decay,
            num_epochs,
            train_set.X, train_set.y,
            valid_set.X, valid_set.y,
            test_set.X, test_set.y,
            save_path = "./cifar10BestModel.pkl",
            shuffle_parts=shuffle_parts)


def build_network (activation, alpha, epsilon, input):
    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    # 128C3-128C3-P2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
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
    # 256C3-256C3-P2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
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
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
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
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
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
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
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
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10)
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    return cnn


def load_dataset (train_set_size):
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
    
    return train_set, valid_set, test_set


if __name__ == "__main__":
    main()
