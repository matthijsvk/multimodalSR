from __future__ import print_function

import os
import time

import lasagne
import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T

from tqdm import tqdm
import math

from general_tools import *


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Helper function that returns an iterator over the training data of a particular
    size, optionally in a random order.

    For big data sets you can load numpy arrays as memory-mapped files
        (numpy.load(..., mmap_mode='r'))

    This function a slight modification of:
        http://lasagne.readthedocs.org/en/latest/user/tutorial.html
    """
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            # excerpt = slice(start_idx, start_idx + batch_size)
            excerpt = range(start_idx, start_idx + batch_size, 1)

        input_iter = [inputs[i] for i in excerpt]
        target_iter = [targets[i] for i in excerpt]
        yield input_iter, target_iter
        # yield inputs[excerpt], targets[excerpt]


class NeuralNetwork:
    network = None
    training_fn = None
    best_param = None
    best_error = 100
    curr_epoch, best_epoch = 0, 0

    network_train_info = [[], [], []]

    # [[Train], [val], [test]]

    def build_RNN(self, batch_size=1, num_features=26, n_hidden=275, num_output_units=61,
                  weight_init=0.1, activation_fn=lasagne.nonlinearities.sigmoid,
                  seed=int(time.time()), debug=False):
        np.random.seed(seed)
        # seed np for weight initialization

        l_in = L.InputLayer(shape=(batch_size, None, num_features))
        #l_in = L.InputLayer(shape=(None, None, num_features))      #compile for variable batch size; slower
        # (batch_size, max_time_steps, n_features_1, n_features_2, ...)
        # Only stochastic gradient descent

        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (n_batch, n_time_steps);
        # both of these dimensions are variable for us so we will use
        # an input shape of (None, None)
        l_mask = lasagne.layers.InputLayer(shape=(None, None))  # See http://colinraffel.com/talks/hammer2015recurrent.pdf

        if debug:
            print('  output size: ', end='\t');
            print(self.Y.shape)
            print(self.Y[0].shape, type(self.Y[0]), type(self.Y[0][0]), self.Y[0])

            print('  input size:', end='\t');
            print(self.X[0].shape)
            print(self.X[0][0].shape,type(self.X[0][0]), type(self.X[0][0][0]), self.X[0][0])

            #get_l_in = theano.function([l_in.input_var], L.get_output(l_in))
            get_l_in = L.get_output(l_in)
            l_in_val = get_l_in.eval({l_in.input_var: self.X})
            #l_in_val = get_l_in(self.X)
            print(get_l_in)
            print(l_in_val)
            print('  l_in size:', end='\t');
            print(l_in_val.shape)

        l_rnn = L.recurrent.RecurrentLayer(
                l_in, num_units=n_hidden,
                nonlinearity=activation_fn,
                W_in_to_hid=lasagne.init.Uniform(weight_init),
                W_hid_to_hid=lasagne.init.Uniform(weight_init),
                b=lasagne.init.Constant(0.),
                hid_init=lasagne.init.Constant(0.),
                learn_init=False)

        # ## LSTM stuff
        # # All gates have initializers for the input-to-gate and hidden state-to-gate
        # # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
        # # The convention is that gates use the standard sigmoid nonlinearity,
        # # which is the default for the Gate class.
        # gate_parameters = lasagne.layers.recurrent.Gate(
        #         W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        #         b=lasagne.init.Constant(0.))
        # cell_parameters = lasagne.layers.recurrent.Gate(
        #         W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        #         # Setting W_cell to None denotes that no cell connection will be used.
        #         W_cell=None, b=lasagne.init.Constant(0.),
        #         # By convention, the cell nonlinearity is tanh in an LSTM.
        #         nonlinearity=lasagne.nonlinearities.tanh)
        # # Our LSTM will have 10 hidden/cell units
        # N_HIDDEN = 10
        # l_lstm = lasagne.layers.recurrent.LSTMLayer(
        #         l_in, N_HIDDEN,
        #         # We need to specify a separate input for masks
        #         mask_input=l_mask,
        #         # Here, we supply the gate parameters for each gate
        #         ingate=gate_parameters, forgetgate=gate_parameters,
        #         cell=cell_parameters, outgate=gate_parameters,
        #         # We'll learn the initialization and use gradient clipping
        #         learn_init=True, grad_clipping=100.)

        # Bidirectional: add reverse layer
        # # The "backwards" layer is the same as the first,
        # # except that the backwards argument is set to True.
        # l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
        #         l_in, N_HIDDEN, ingate=gate_parameters,
        #         mask_input=l_mask, forgetgate=gate_parameters,
        #         cell=cell_parameters, outgate=gate_parameters,
        #         learn_init=True, grad_clipping=100., backwards=True)

        # # We'll combine the forward and backward layer output by summing.
        # # Merge layers take in lists of layers to merge as input.
        # l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])
        # # The output of l_msum will be of shape (n_batch, n_time_steps, num_features).

        # # # Now we need to go from RNN to Feedforward -> different shape expected -> reshape
        # # First, retrieve symbolic variables for the input shape
        # n_batch, n_time_steps, n_features = l_in.input_var.shape
        # # Now, squash the n_batch and n_time_steps dimensions
        # l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))
        # # Now, we can apply feed-forward layers as usual.
        # # We want the network to predict a single value, the sum, so we'll use a single unit.
        # l_dense = lasagne.layers.DenseLayer(
        #         l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
        # # Now, the shape will be n_batch*n_timesteps, 1. We can then reshape to
        # # n_batch, n_timesteps to get a single value for each timstep from each sequence
        # l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_time_steps))
        if debug:
            get_l_rnn = theano.function([l_in.input_var], L.get_output(l_rnn))
            l_rnn_val = get_l_rnn(self.X)
            print('  l_rnn size:', end='\t');
            print(l_rnn_val.shape)

        l_reshape = L.ReshapeLayer(l_rnn, (-1, n_hidden))
        if debug:
            get_l_reshape = theano.function([l_in.input_var], L.get_output(l_reshape))
            l_reshape_val = get_l_reshape(self.X)
            print('  l_reshape size:', end='\t')
            print(l_reshape_val.shape)

        l_out = L.DenseLayer(l_reshape, num_units=num_output_units,
                             nonlinearity=T.nnet.softmax)

        self.network = l_out

    def __init__(self, architecture, dataset, **kwargs):
        if architecture == 'RNN':
            X_train, y_train, X_val, y_val, X_test, y_test = dataset
            self.X = X_train
            self.Y = y_train
            self.build_RNN(**kwargs)
        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")



    def use_best_param(self):
        lasagne.layers.set_all_param_values(self.network, self.best_param)
        self.curr_epoch = self.best_epoch
        # Remove the network_train_info enries newer than self.best_epoch
        del self.network_train_info[0][self.best_epoch:]
        del self.network_train_info[1][self.best_epoch:]
        del self.network_train_info[2][self.best_epoch:]

    def load_model(self, model_name):
        if self.network is not None:
            try:
                print("Loading previous model...")
                with np.load(model_name) as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                # param_values[0] = param_values[0].astype('float32')
                param_values = [param_values[i].astype('float32') for i in range(len(param_values))]
                lasagne.layers.set_all_param_values(self.network, param_values)
            except IOError as e:
                #print(os.strerror(e.errno))
                print('Model: {} not found. No weights loaded'.format(model_name))
        else:
            print('You must build the network before loading the weights.')

    def save_model(self, model_name):
        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))
        np.savez(model_name, *L.get_all_param_values(self.network))

    def build_functions(self, LEARNING_RATE=1e-5, MOMENTUM=0.9, debug=False):

        target_var = T.ivector('targets')

        network_output = L.get_output(self.network)

        # Retrieve all trainable parameters from the network
        all_params = L.get_all_params(self.network, trainable=True)

        # cost = T.mean(lasagne.objectives.categorical_crossentropy(network_output, target_var))
        cost = T.sum(lasagne.objectives.categorical_crossentropy(network_output, target_var))

        # Function to determine the number of correct classifications
        accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), target_var),
                          dtype=theano.config.floatX)

        # use Stochastic Gradient Descent with nesterov momentum to update parameters
        # updates = lasagne.updates.momentum(cost, all_params,
        #                                    learning_rate=LEARNING_RATE,
        #                                    momentum=MOMENTUM)
        updates = lasagne.updates.adam(cost, all_params)

        # Get the first layer of the network
        l_in = L.get_all_layers(self.network)[0]

        # Function to get the output of the network
        output_fn = theano.function([l_in.input_var], network_output, name='output_fn')
        if debug:
            print(self.X.shape)
            x = self.X.shape[0]; y = self.X[0].shape[0]; z = self.X[0].shape[1]
            print(x,y,z)
            self.X = np.reshape(self.X, (x,y,z))
            print(l_in.input_var.type)
            l_out_val = output_fn(self.X)
            print('l_out size:', end='\t');
            print(l_out_val.shape, end='\t');
            print('min/max: [{:.2f},{:.2f}]'.format(l_out_val.min(), l_out_val.max()))

        argmax_fn = theano.function([l_in.input_var], [T.argmax(network_output, axis=1)],
                                    name='argmax_fn')
        if debug:
            print('argmax_fn')
            print(type(argmax_fn(self.X)[0]))
            print(argmax_fn(self.X)[0].shape)

        # Function implementing one step of gradient descent
        train_fn = theano.function([l_in.input_var, target_var], [cost, accuracy],
                                   updates=updates, name='train_fn')

        # Function calculating the cost and accuracy
        validate_fn = theano.function([l_in.input_var, target_var], [cost, accuracy],
                                      name='validate_fn')

        if debug:
            print(type(train_fn(self.X, self.Y)))
            print('cost: {:.3f}'.format( float(train_fn(self.X, self.Y))))
            print('accuracy: {:.3f}'.format( float(validate_fn(self.X, self.Y)[1]) ))

        self.training_fn = output_fn, argmax_fn, train_fn, validate_fn

    def create_confusion(self, X, y, debug=False):
        argmax_fn = self.training_fn[1]

        y_pred = []
        for X_obs in X:
            for x in argmax_fn(X_obs):
                for j in x:
                    y_pred.append(j)

        y_actu = []
        for Y in y:
            for y in Y:
                y_actu.append(y)

        conf_img = np.zeros([61, 61])
        assert (len(y_pred) == len(y_actu))

        for i in range(len(y_pred)):
            row_idx = y_actu[i]
            col_idx = y_pred[i]
            conf_img[row_idx, col_idx] += 1

        return conf_img, y_pred, y_actu

    def create_learning_curves(self):
        pass

    def visualize_training(self, learning_curves, confusion):
        pass

    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1,
              compute_confusion=False, debug=False):
        """Curently one batch_size=1 is supported"""

        X_train, y_train, X_val, y_val, X_test, y_test = dataset
        output_fn, argmax_fn, train_fn, validate_fn = self.training_fn

        if debug:
            print('X_train')
            print(type(X_train), len(X_train))
            print('X_train[0]', type(X_train[0]), X_train[0].shape)
            print('X_train[0][0]', type(X_train[0][0]), X_train[0][0].shape)
            print('X_train[0][0][0]', type(X_train[0][0][0]), X_train[0][0][0].shape)
            print('y_train')
            print('y_train', type(y_train), len(y_train))
            print('y_train[0]', type(y_train[0]), y_train[0].shape)
            print('y_train[0][0]', type(y_train[0][0]), y_train[0][0].shape)

        # Initiate some vectors used for tracking performance
        train_error = np.zeros([num_epochs])
        train_accuracy = np.zeros([num_epochs])
        train_batches = np.zeros([num_epochs])

        validation_error = np.zeros([num_epochs])
        validation_accuracy = np.zeros([num_epochs])
        validation_batches = np.zeros([num_epochs])

        test_error = np.zeros([num_epochs])
        test_accuracy = np.zeros([num_epochs])
        test_batches = np.zeros([num_epochs])

        confusion_matrices = []

        for epoch in range(num_epochs):
            self.curr_epoch += 1
            epoch_time = time.time()

            # Full pass over the training set
            for inputs, targets in tqdm(iterate_minibatches(X_train, y_train, batch_size, shuffle=True), total=math.ceil(len(X_train)/batch_size)):
                for i in range(len(inputs)):
                    # TODO: this for loop should not exist

                    # if debug:
                    #     print(type(inputs), type(targets))
                    #     print(type(inputs[i]), type(targets[i]))
                    error, accuracy = train_fn([inputs[i]], targets[i])

                    train_error[epoch] += error
                    train_accuracy[epoch] += accuracy
                    train_batches[epoch] += 1

            # Full pass over the validation set
            for inputs, targets in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                for i in range(len(inputs)):
                    error, accuracy = validate_fn([inputs[i]], targets[i])

                    validation_error[epoch] += error
                    validation_accuracy[epoch] += accuracy
                    validation_batches[epoch] += 1

            # Full pass over the test set
            for inputs, targets in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                for i in range(len(inputs)):
                    error, accuracy = validate_fn([inputs[i]], targets[i])

                    test_error[epoch] += error
                    test_accuracy[epoch] += accuracy
                    test_batches[epoch] += 1

            # Print epoch summary
            train_epoch_error = (100 - train_accuracy[epoch]
                                 / train_batches[epoch] * 100)
            val_epoch_error = (100 - validation_accuracy[epoch]
                               / validation_batches[epoch] * 100)
            test_epoch_error = (100 - test_accuracy[epoch]
                                / test_batches[epoch] * 100)

            self.network_train_info[0].append(train_epoch_error)
            self.network_train_info[1].append(val_epoch_error)
            self.network_train_info[2].append(test_epoch_error)

            print("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time), end=' ')
            if val_epoch_error < self.best_error:
                self.best_error = val_epoch_error
                self.best_epoch = self.curr_epoch
                self.best_param = L.get_all_param_values(self.network)
                print("New best model found!")
            else:
                print()

            print("  New best model found!")
            if save_name is not None:
                print("Model saved as " + save_name + '.npz')
                self.save_model(save_name + '.npz')
            else:
                print()

            print("  training cost:\t{:.6f}".format(
                    train_error[epoch] / train_batches[epoch]))
            print("train error:\t\t{:.6f} %".format(train_epoch_error))

            print("  validation cost:\t{:.6f}".format(
                    validation_error[epoch] / validation_batches[epoch]))
            print("validation error:\t{:.6f} %".format(val_epoch_error))

            print("  test cost:\t\t{:.6f}".format(
                    test_error[epoch] / test_batches[epoch]))
            print("test error:\t\t{:.6f} %".format(test_epoch_error))

            if compute_confusion:
                confusion_matrices.append(self.create_confusion(X_val, y_val)[0])
                print('  Confusion matrix computed')
            
            print()

            with open(save_name + '_var.pkl', 'wb') as cPickle_file:
                cPickle.dump(
                        [self.network_train_info],
                        cPickle_file,
                        protocol=cPickle.HIGHEST_PROTOCOL)

            if compute_confusion:
                with open(save_name + '_conf.pkl', 'wb') as cPickle_file:
                    cPickle.dump(
                            [confusion_matrices],
                            cPickle_file,
                            protocol=cPickle.HIGHEST_PROTOCOL)
