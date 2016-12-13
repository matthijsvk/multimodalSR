# ##################### Build the neural network model #######################
from __future__ import print_function

import numpy as np

np.random.seed(1234)  # for reproducibility?

import os

os.environ["THEANO_FLAGS"] = "cuda.root=/usr/local/cuda,device=gpu,floatX=float32"
import theano
import theano.tensor as T
import lasagne

# specifying the gpu to use
import theano.sandbox.cuda

theano.sandbox.cuda.use('gpu1')

import train_lipreadingTCDTIMIT  # load training functions
from loadData import CIFAR10  # load the binary dataset in proper format
from resnet50 import *

# from http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from lasagne.layers import Conv2DLayer as ConvLayer

# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm


def main ():
    # BN parameters
    batch_size = 50
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
    
    train_set_size = 11500
    print("train_set_size = " + str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))
    
    print('Loading TCDTIMIT dataset...')
    train_set, valid_set, test_set = load_dataset(train_set_size)
    
    print("the number of training examples is: ", len(train_set.X))
    print("the number of valid examples is: ", len(valid_set.X))
    print("the number of test examples is: ", len(test_set.X))
    
    print('Building the CNN...')
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    # get the network structure
    # cnn = build_network_google(activation, alpha, epsilon, input)
    # cnn = build_network_cifar10(activation, alpha, epsilon, input)
    
    # get output layer, for calculating loss etc
    # train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # resnet50; needs to be evaluated differently as well -> comment above line
    cnn = build_network_resnet(input)
    print("number of parameters in model: %d" % lasagne.layers.count_params(cnn, trainable=True))
    prediction = lasagne.layers.get_output(cnn['prob'])
    
    # update loss
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss = loss + l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
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
            train_set.X, train_set.y,
            valid_set.X, valid_set.y,
            test_set.X, test_set.y,
            save_path="./TCDTIMITBestModel.pkl",
            shuffle_parts=shuffle_parts)

# from https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
def build_network_resnet (input_var=None, n=5):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block (l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters
        
        stack_1 = batch_norm(
                ConvLayer(l, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride, nonlinearity=rectify,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
                ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                        ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                                  pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)
        
        return block
    
    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    
    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad='same',
                             W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)
    
    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)
    
    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)
    
    # average pooling
    l = GlobalPoolLayer(l)
    
    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)
    
    return network


#############################3
# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def load_dataset (train_set_size):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    # also see http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10
    
    # CIFAR10 files stored in /home/matthijs/Documents/Pylearn_datasets/cifar10/cifar-10-batches-py
    # then processed with /home/matthijs/bin/pylearn2/pylearn2/datasets/cifar10.py
    
    # our files are stored in /home/matthijs/TCDTIMIT/database_binary
    # and processed with ./loadData.py
    
    # Lipspeaker 1:                  14627 phonemes, apparently only 14530 extracted
    # Lipspeaker 2:  28363 - 14627 = 13736 phonemes
    # Lipspeaker 3:  42535 - 28363 = 14172 phonemes
    
    # lipspeaker 1 : 14627 -> 11.5k train, 1.5k valid, 1.627k test
    train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
    valid_set = CIFAR10(which_set="train", start=train_set_size, stop=13000)
    test_set = CIFAR10(which_set="test")
    
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_set.X = np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 1, 120, 120))
    valid_set.X = np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 1, 120, 120))
    test_set.X = np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 1, 120, 120))
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


if __name__ == "__main__":
    main()
