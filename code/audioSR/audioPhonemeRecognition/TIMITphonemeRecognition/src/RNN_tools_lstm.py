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

import pdb
import logging  # debug < info < warn < error < critical  # from https://docs.python.org/3/howto/logging-cookbook.html

logger_RNNtools = logging.getLogger('RNNtools')
logger_RNNtools.setLevel(logging.DEBUG)

from general_tools import *


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Helper function that returns an iterator over the training data of a particular
    size, optionally in a random order.
    """
    assert len(inputs) == len(targets)
    if len(inputs) < batch_size:
        batch_size = len(inputs)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = range(start_idx, start_idx + batch_size, 1)

        input_iter = [inputs[i] for i in excerpt]
        target_iter = [targets[i] for i in excerpt]
        mask_iter = generate_masks(input_iter, batch_size)
        seq_lengths = np.sum(mask_iter, axis=1)

        # now pad inputs and target to maxLen
        input_iter = pad_sequences_X(input_iter)
        target_iter = pad_sequences_y(target_iter)

        yield input_iter, target_iter, mask_iter, seq_lengths
        #  it's convention that data is presented in the shape (batch_size, n_time_steps, n_features) -> (batch_size, None, 26)


class NeuralNetwork:
    network = None
    training_fn = None
    best_param = None
    best_error = 100
    curr_epoch, best_epoch = 0, 0
    X = None
    Y = None

    network_train_info = [[], [], []]

    def __init__(self, architecture, dataset, batch_size=1, num_features=26, n_hidden=275, num_output_units=61,
                 weight_init=0.1, activation_fn=lasagne.nonlinearities.sigmoid,
                 seed=int(time.time()), debug=False):
        if architecture == 'RNN':
            X_train, y_train, X_val, y_val, X_test, y_test = dataset

            X = X_train[:batch_size]
            y = y_train[:batch_size]
            self.masks = generate_masks(X, len(X))

            self.X = pad_sequences_X(X)
            self.Y = pad_sequences_y(y)

            logger_RNNtools.debug('X.shape:          %s', self.X.shape)
            logger_RNNtools.debug('X[0].shape:       %s', self.X[0].shape)
            logger_RNNtools.debug('X[0][0][0].type:  %s', type(self.X[0][0][0]))
            logger_RNNtools.debug('y.shape:          %s', self.Y.shape)
            logger_RNNtools.debug('y[0].shape:       %s', self.Y[0].shape)
            logger_RNNtools.debug('y[0][0].type:     %s', type(self.Y[0][0]))
            logger_RNNtools.debug('masks.shape:      %s', self.masks.shape)
            logger_RNNtools.debug('masks[0].shape:   %s', self.masks[0].shape)
            logger_RNNtools.debug('masks[0][0].type: %s', type(self.masks[0][0]))

            self.build_RNN(batch_size, num_features, n_hidden, num_output_units,
                           weight_init, activation_fn, seed, debug)
        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")

    def build_RNN(self, batch_size=1, num_features=26, n_hidden=275, num_output_units=61,
                  weight_init=0.1, activation_fn=lasagne.nonlinearities.sigmoid,
                  seed=int(time.time()), debug=False):
        if debug:
            logger_RNNtools.debug('\nInputs:');
            logger_RNNtools.debug('  X.shape:    %s', self.X[0].shape)
            logger_RNNtools.debug('  X[0].shape: %s %s %s \n%s', self.X[0][0].shape, type(self.X[0][0]),
                                  type(self.X[0][0][0]), self.X[0][0][:5])

            logger_RNNtools.debug('Targets: ');
            logger_RNNtools.debug('  Y.shape:    %s', self.Y.shape)
            logger_RNNtools.debug('  Y[0].shape: %s %s %s \n%s', self.Y[0].shape, type(self.Y[0]), type(self.Y[0][0]),
                                  self.Y[0][:5])
            logger_RNNtools.debug('Layers: ')

        np.random.seed(seed)
        # seed np for weight initialization
        net = {}

        max_seq_length = 1000
        # shape = (batch_size, max_seq_length, num_features), but 1 and 2 are variable
        net['l1_in'] = L.InputLayer(shape=(batch_size, None, num_features))
        # l_in = L.InputLayer(shape=(None, None, num_features))      #compile for variable batch size; slower

        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (n_batch, n_time_steps);
        net['l1_mask'] = L.InputLayer(shape=(batch_size, None))  # See http://colinraffel.com/talks/hammer2015recurrent.pdf

        if debug:
            get_l_in = L.get_output(net['l1_in'])
            l_in_val = get_l_in.eval({net['l1_in'].input_var: self.X})
            #logger_RNNtools.debug(l_in_val)
            logger_RNNtools.debug('  l_in size: %s', l_in_val.shape);

            get_l_mask = L.get_output(net['l1_mask'])
            l_mask_val = get_l_mask.eval({net['l1_mask'].input_var: self.masks})
            # logger_RNNtools.debug(l_in_val)
            logger_RNNtools.debug('  l_mask size: %s', l_mask_val.shape);

            n_batch, n_time_steps, n_features = net['l1_in'].input_var.shape
            logger_RNNtools.debug("  n_batch: %s | n_time_steps: %s | n_features: %s", n_batch, n_time_steps, n_features)


        ## LSTM parameters
        # All gates have initializers for the input-to-gate and hidden state-to-gate
        # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
        # The convention is that gates use the standard sigmoid nonlinearity,
        # which is the default for the Gate class.
        gate_parameters = lasagne.layers.recurrent.Gate(
                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                b=lasagne.init.Constant(0.))
        cell_parameters = lasagne.layers.recurrent.Gate(
                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                # Setting W_cell to None denotes that no cell connection will be used.
                W_cell=None, b=lasagne.init.Constant(0.),
                # By convention, the cell nonlinearity is tanh in an LSTM.
                nonlinearity=lasagne.nonlinearities.tanh)

        net['l2_lstm'] = lasagne.layers.recurrent.LSTMLayer(
                net['l1_in'], n_hidden,
                # We need to specify a separate input for masks
                mask_input=net['l1_mask'],
                # Here, we supply the gate parameters for each gate
                ingate=gate_parameters, forgetgate=gate_parameters,
                cell=cell_parameters, outgate=gate_parameters,
                # We'll learn the initialization and use gradient clipping
                learn_init=True, grad_clipping=100.)

        # The "backwards" layer is the same as the first,
        # except that the backwards argument is set to True.
        net['l3_lstm_back'] = lasagne.layers.recurrent.LSTMLayer(
                net['l1_in'], n_hidden, ingate=gate_parameters,
                mask_input=net['l1_mask'], forgetgate=gate_parameters,
                cell=cell_parameters, outgate=gate_parameters,
                learn_init=True, grad_clipping=100., backwards=True)
        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        # The output of l_sum will be of shape (n_batch, max_n_time_steps, n_features)
        net['l4_lsum']= lasagne.layers.ElemwiseSumLayer([net['l2_lstm'], net['l3_lstm_back']])

        # To connect this to feedforward networks, we need to reshape
        # use symbolic variables from the input shape

        # Now, squash the n_batch and n_time_steps dimensions
        net['l5_reshape'] = lasagne.layers.ReshapeLayer(net['l4_lsum'], (-1, n_hidden))

        # Now, we can apply feed-forward layers as usual for classification
        net['l6_dense'] = L.DenseLayer(net['l5_reshape'], num_units=num_output_units,
                                    nonlinearity=T.nnet.softmax)
        # Now, the shape will be (n_batch * n_timesteps, num_output_units. We can then reshape to
        # n_batch to get num_output_units values for each timestep from each sequence
        net['l7_out'] = lasagne.layers.ReshapeLayer(net['l6_dense'], (-1, num_output_units))

        if debug:
            # Forwards LSTM
            get_l_lstm = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var], L.get_output(net['l2_lstm']))
            l_lstm_val = get_l_lstm(self.X, self.masks)
            logger_RNNtools.debug('  l2_lstm size: %s', l_lstm_val.shape);
            # Backwards LSTM
            get_l_lstm_back = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var], L.get_output(net['l3_lstm_back']))
            l_lstmBack_val = get_l_lstm_back(self.X, self.masks)
            logger_RNNtools.debug('  l3_lstm_back size: %s', l_lstmBack_val.shape)

        if debug:
            get_l_reshape = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var], L.get_output(net['l5_reshape']))
            l_reshape_val = get_l_reshape(self.X, self.masks)
            logger_RNNtools.debug('  l_reshape size: %s', l_reshape_val.shape)

            # print network structure
            logger_RNNtools.debug("\n PRINTING Network structure: \n %s ", sorted(net.keys()))
            for key in sorted(net.keys()):
                try:logger_RNNtools.debug('Layer: %12s | in: %s | out: %s', key, net[key].input_shape, net[key].output_shape)
                except: logger_RNNtools.debug('Layer: %12s | out: %s', key, net[key].output_shape)

            pdb.set_trace()
        self.network = net



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
                #print("Loading previous model...")
                with np.load(model_name) as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                # param_values[0] = param_values[0].astype('float32')
                param_values = [param_values[i].astype('float32') for i in range(len(param_values))]
                lasagne.layers.set_all_param_values(self.network, param_values)
            except IOError as e:
                # print(os.strerror(e.errno))
                logger_RNNtools.warning('Model: {} not found. No weights loaded'.format(model_name))
        else:
            logger_RNNtools.error('You must build the network before loading the weights.')
            raise Exception

    def save_model(self, model_name):
        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))
        np.savez(model_name, self.best_param)

    def build_functions(self, LEARNING_RATE=1e-5, MOMENTUM=0.9, debug=False):  # LSTM in lasagne: see https://github.com/craffel/Lasagne-tutorial/blob/master/examples/recurrent.py

        target_var = T.imatrix('targets')

        net_out = self.network['l7_out']
        network_output = L.get_output(net_out)

        # Get the first layer of the network
        l_in = self.network['l1_in']
        l_mask = self.network['l1_mask']

        if debug:
            # print network structure
            net = self.network
            logger_RNNtools.debug("\n PRINTING Network structure: \n %s ", sorted(net.keys()))
            for key in sorted(net.keys()):
                try:
                    logger_RNNtools.debug('Layer: %12s | in: %s | out: %s', key, net[key].input_shape,
                                          net[key].output_shape)
                except:
                    logger_RNNtools.debug('Layer: %12s | out: %s', key, net[key].output_shape)

        # Function to get the output of the network
        output_fn = theano.function([l_in.input_var, l_mask.input_var], network_output, name='output_fn')
        if debug:
            logger_RNNtools.debug('l_in.input_var.type: \t%s', l_in.input_var.type)
            logger_RNNtools.debug('l_in.input_var.shape:\t%s', l_in.input_var.shape)

            l_out_val = output_fn(self.X, self.masks)
            logger_RNNtools.debug('output_fn(X)[0]:     \n%s', l_out_val[0]);
            logger_RNNtools.debug('output_fn(X), shape: \t%s', l_out_val.shape);
            logger_RNNtools.debug(
                'output_fn(X), min/max: [{:.2f},{:.2f}]'.format(l_out_val.min(), l_out_val.max()))


        # Retrieve all trainable parameters from the network
        all_params = L.get_all_params(net_out, trainable=True)

        # compare targets with highest output probability. Take maximum of all probs (3rd axis of output: 1=batch_size (input files), 2 = time_seq (frames), 3 = n_features (phonemes)
        # network_output.shape = (len(X), 39) -> (nb_inputs, nb_classes)

        ## from https://groups.google.com/forum/#!topic/lasagne-users/os0j3f_Th5Q
        # Pad your vector of labels and then mask the cost:
        # cost = lasagne.objectives.categorical_crossentropy(predictions, targets)
        # cost = lasagne.objectives.aggregate(cost, mask.flatten())
        # The latter will do (cost * mask).mean().
        # It's important to pad the label vectors with something valid such as zeros,
        # since they will still have to give valid costs that can be multiplied by the mask.
        # The shape of predictions, targets and mask should match:
        # (predictions as (batch_size*max_seq_len, n_features), the other two as (batch_size*max_seq_len,)).
        cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output, target_var.flatten())
        cost = (cost_pointwise * l_mask.input_var.flatten()).mean()

        # Function to determine the number of correct classifications
        predictions = (T.argmax(network_output, axis=1))
        predictions_fn = theano.function([l_in.input_var, l_mask.input_var], predictions, name='predictions_fn')
        if debug:
            predicted = predictions_fn(self.X, self.masks)
            logger_RNNtools.debug('predictions_fn(X).shape: %s', predicted.shape)
            logger_RNNtools.debug('predictions_fn(X)[0], value: \n%s', predicted[0])

        # TODO: only use the output at the middle of each phoneme interval (get better accuracy)
        accuracy = T.mean(T.eq(predictions, target_var.flatten()), dtype=theano.config.floatX)


        # Functions for computing cost and training
        validate_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                      [cost, accuracy], name='validate_fn')
        cost_pointwise_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                            cost_pointwise, name='cost_pointwise_fn')
        if debug:
            logger_RNNtools.debug('%s', self.Y.flatten())

            logger_RNNtools.debug('%s', cost_pointwise_fn(self.X, self.masks, self.Y))

            evaluate_cost = validate_fn(self.X, self.masks, self.Y)
            logger_RNNtools.debug('%s %s', type(evaluate_cost), len(evaluate_cost))
            logger_RNNtools.debug('%s', evaluate_cost)
            logger_RNNtools.debug('cost:     {:.3f}'.format(float(evaluate_cost[0])))
            logger_RNNtools.debug('accuracy: {:.3f}'.format(float(evaluate_cost[1])))

        #pdb.set_trace()

        updates = lasagne.updates.adam(cost, all_params)
        train_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                [cost, accuracy], updates=updates, name='train_fn')


        self.training_fn = output_fn, predictions_fn, train_fn, validate_fn

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


    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1,
              compute_confusion=False, debug=False):

        X_train, y_train, X_val, y_val, X_test, y_test = dataset
        output_fn, argmax_fn, train_fn, validate_fn = self.training_fn

        if debug:
            logger_RNNtools.debug('  X_train')
            logger_RNNtools.debug('%s %s', type(X_train), len(X_train))
            logger_RNNtools.debug('X_train[0] %s %s', type(X_train[0]), X_train[0].shape)
            logger_RNNtools.debug('X_train[0][0] %s %s', type(X_train[0][0]), X_train[0][0].shape)
            logger_RNNtools.debug('X_train[0][0][0]  %s %s', type(X_train[0][0][0]), X_train[0][0][0].shape)
            logger_RNNtools.debug('  y_train')
            logger_RNNtools.debug('y_train  %s %s', type(y_train), len(y_train))
            logger_RNNtools.debug('y_train[0]  %s %s', type(y_train[0]), y_train[0].shape)
            logger_RNNtools.debug('y_train[0][0]  %s %s', type(y_train[0][0]), y_train[0][0].shape)

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

        logger_RNNtools.info("\n* Starting training...")
        for epoch in range(num_epochs):
            self.curr_epoch += 1
            epoch_time = time.time()

            logger_RNNtools.info("CURRENT EPOCH: %s", self.curr_epoch)

            logger_RNNtools.info("Pass over Training Set")
            for inputs, targets, masks, seq_lengths in tqdm(iterate_minibatches(X_train, y_train, batch_size, shuffle=True),
                                        total=math.ceil(len(X_train) / batch_size)):

                if debug:
                    logger_RNNtools.debug('%s %s',inputs.shape, targets.shape)
                    logger_RNNtools.debug('%s %s',inputs[0].shape, targets[0].shape)
                error, accuracy = train_fn(inputs, masks, targets)
                if debug: logger_RNNtools.debug('%s %s', error, accuracy)
                train_error[epoch] += error
                train_accuracy[epoch] += accuracy
                train_batches[epoch] += len(inputs)
                #pdb.set_trace()

            logger_RNNtools.info("Pass over Validation Set")
            for inputs, targets, masks, seq_lengths in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                    error, accuracy = validate_fn(inputs, masks, targets)
                    validation_error[epoch] += error
                    validation_accuracy[epoch] += accuracy
                    validation_batches[epoch] += len(inputs)

            logger_RNNtools.info("Pass over Test Set")
            for inputs, targets, masks, seq_lengths in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                    error, accuracy = validate_fn(inputs, masks, targets)
                    test_error[epoch] += error
                    test_accuracy[epoch] += accuracy
                    test_batches[epoch] += len(inputs)

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

            logger_RNNtools.info("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time))
            if val_epoch_error < self.best_error:
                self.best_error = val_epoch_error
                self.best_epoch = self.curr_epoch
                self.best_param = L.get_all_param_values(self.network['l7_out'])
                logger_RNNtools.info("New best model found!")
                if save_name is not None:
                    logger_RNNtools.info("Model saved as " + save_name + '.npz')
                    self.save_model(save_name + '.npz')


            logger_RNNtools.info("training cost:\t{:.6f}".format(
                    train_error[epoch] / train_batches[epoch]))
            logger_RNNtools.info("  train error:\t\t{:.6f} %".format(train_epoch_error))

            logger_RNNtools.info("validation cost:\t{:.6f}".format(
                    validation_error[epoch] / validation_batches[epoch]))
            logger_RNNtools.info("  validation error:\t{:.6f} %".format(val_epoch_error))

            logger_RNNtools.info("test cost:\t\t{:.6f}".format(
                    test_error[epoch] / test_batches[epoch]))
            logger_RNNtools.info("  test error:\t\t{:.6f} %".format(test_epoch_error))

            if compute_confusion:
                confusion_matrices.append(self.create_confusion(X_val, y_val)[0])
                logger_RNNtools.info('  Confusion matrix computed')


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
                            