from __future__ import print_function

import logging  # debug < info < warn < error < critical  # from https://docs.python.org/3/howto/logging-cookbook.html
import math
import os
import time
import traceback

import lasagne
import lasagne.layers as L
import theano
import theano.tensor as T
from tqdm import tqdm

logger_RNNtools = logging.getLogger('RNN.tools')
logger_RNNtools.setLevel(logging.DEBUG)

from general_tools import *


def iterate_minibatches(inputs, targets, valid_frames, batch_size, shuffle=False):
    """
    Helper function that returns an iterator over the training data of a particular
    size, optionally in a random order.
    """
    assert len(inputs) == len(targets) == len(valid_frames)
    if len(inputs) < batch_size:
        batch_size = len(inputs)

    # slice to only use multiple of batch_size. If some files are left, they won't be considered
    # inputs = inputs[:-(len(inputs) % batch_size) or None]
    # targets = targets[:-(len(targets) % batch_size) or None]
    # valid_frames = valid_frames[:-(len(valid_frames) % batch_size) or None]

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
        valid_frames_iter = [valid_frames[i] for i in excerpt]
        mask_iter = generate_masks(input_iter, valid_frames=valid_frames_iter, batch_size=batch_size)

# line 145, in generate_masks
#     input_mask[example_id,valid_frames[example_id]] = 1
# IndexError: index 677 is out of bounds for axis 1 with size 677


        seq_lengths = np.sum(mask_iter, axis=1)

        # now pad inputs and target to maxLen
        input_iter = pad_sequences_X(input_iter)
        target_iter = pad_sequences_y(target_iter)

        yield input_iter, target_iter, mask_iter, seq_lengths
        #  it's convention that data is presented in the shape (batch_size, n_time_steps, n_features) -> (batch_size, None, 26)

# used for evaluating, when there are no targets
def iterate_minibatches_noTargets(inputs, valid_frames, batch_size=1, shuffle=False):
    """
    Helper function that returns an iterator over the training data of a particular
    size, optionally in a random order.
    """
    if len(inputs) < batch_size:
        batch_size = len(inputs)
        print("INPUTS < Batch_size")

    # slice to only use multiple of batch_size. If some files are left, they won't be considered

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = range(start_idx, start_idx + batch_size, 1)

        input_iter = [inputs[i] for i in excerpt]
        mask_iter = generate_masks(input_iter, valid_frames=valid_frames, batch_size = batch_size)
        seq_lengths = np.sum(mask_iter, axis=1)

        # now pad inputs and target to maxLen
        input_iter = pad_sequences_X(input_iter)

        yield input_iter, mask_iter, seq_lengths


class NeuralNetwork:
    network = None
    training_fn = None
    best_param = None
    best_error = 100
    curr_epoch, best_epoch = 0, 0
    X = None
    Y = None

    network_train_info = [[], [], []]

    def __init__(self, architecture, dataset=None, batch_size=1, num_features=26, n_hidden_list=(100,), num_output_units=61,
                 bidirectional=False, seed=int(time.time()), debug=False, logger=logger_RNNtools):
        self.num_output_units = num_output_units
        self.num_features = num_features
        self.batch_size = batch_size
        self.epochsNotImproved = 0  #keep track, to now then to stop training
        self.updates = {}

        if architecture == 'RNN':
            if dataset != None:
                X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset

                X = X_train[:batch_size]
                y = y_train[:batch_size]
                valid_frames = valid_frames_train[:batch_size]
                self.masks = generate_masks(X, valid_frames=valid_frames,batch_size=len(X))

                self.X = pad_sequences_X(X)
                self.Y = pad_sequences_y(y)

                logger.debug('X.shape:          %s', self.X.shape)
                logger.debug('X[0].shape:       %s', self.X[0].shape)
                logger.debug('X[0][0][0].type:  %s', type(self.X[0][0][0]))
                logger.debug('y.shape:          %s', self.Y.shape)
                logger.debug('y[0].shape:       %s', self.Y[0].shape)
                logger.debug('y[0][0].type:     %s', type(self.Y[0][0]))
                logger.debug('masks.shape:      %s', self.masks.shape)
                logger.debug('masks[0].shape:   %s', self.masks[0].shape)
                logger.debug('masks[0][0].type: %s', type(self.masks[0][0]))

            logger.info("NUM FEATURES: %s", num_features)

            self.build_RNN(batch_size, num_features, n_hidden_list, num_output_units, bidirectional,
                           seed, debug)
        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")

    def build_RNN(self, batch_size=1, num_features=26, n_hidden_list=(100,), num_output_units=61, bidirectional=False,
                  seed=int(time.time()), debug=False, logger=logger_RNNtools):
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

        # seed np for weight initialization
        np.random.seed(seed)

        net = {}
        #n_hidden = n_hidden_list[0]
        # some inspiration from http://colinraffel.com/talks/hammer2015recurrent.pdf
        # shape = (batch_size, max_seq_length, num_features)
        net['l1_in'] = L.InputLayer(shape=(None, None, num_features))

        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (batch_size, n_time_steps);
        net['l1_mask'] = L.InputLayer(shape=(None, None))

        if debug:
            get_l_in = L.get_output(net['l1_in'])
            l_in_val = get_l_in.eval({net['l1_in'].input_var: self.X})
            # logger_RNNtools.debug(l_in_val)
            logger_RNNtools.debug('  l_in size: %s', l_in_val.shape);

            get_l_mask = L.get_output(net['l1_mask'])
            l_mask_val = get_l_mask.eval({net['l1_mask'].input_var: self.masks})
            # logger_RNNtools.debug(l_in_val)
            logger_RNNtools.debug('  l_mask size: %s', l_mask_val.shape);

            n_batch, n_time_steps, n_features = net['l1_in'].input_var.shape
            logger_RNNtools.debug("  n_batch: %s | n_time_steps: %s | n_features: %s", n_batch, n_time_steps,
                                  n_features)

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

        # generate layers of stacked LSTMs, possibly bidirectional
        net['l2_lstm'] = []

        for i in range(len(n_hidden_list)):
            n_hidden = n_hidden_list[i]

            if i==0: input = net['l1_in']
            else:    input = net['l2_lstm'][i-1]

            nextForwardLSTMLayer = lasagne.layers.recurrent.LSTMLayer(
                    input, n_hidden,
                    # We need to specify a separate input for masks
                    mask_input=net['l1_mask'],
                    # Here, we supply the gate parameters for each gate
                    ingate=gate_parameters, forgetgate=gate_parameters,
                    cell=cell_parameters, outgate=gate_parameters,
                    # We'll learn the initialization and use gradient clipping
                    learn_init=True, grad_clipping=100.)
            net['l2_lstm'].append(nextForwardLSTMLayer)

            if bidirectional:
                input = net['l2_lstm'][-1]
                # Use backward LSTM
                # The "backwards" layer is the same as the first,
                # except that the backwards argument is set to True.
                nextBackwardLSTMLayer = lasagne.layers.recurrent.LSTMLayer(
                        input, n_hidden, ingate=gate_parameters,
                        mask_input=net['l1_mask'], forgetgate=gate_parameters,
                        cell=cell_parameters, outgate=gate_parameters,
                        learn_init=True, grad_clipping=100., backwards=True)
                net['l2_lstm'].append(nextBackwardLSTMLayer)

                if debug:
                    # Backwards LSTM
                    get_l_lstm_back = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                                      L.get_output(net['l2_lstm'][-1]))
                    l_lstmBack_val = get_l_lstm_back(self.X, self.masks)
                    logger_RNNtools.debug('  l_lstm_back size: %s', l_lstmBack_val.shape)

                # We'll combine the forward and backward layer output by summing.
                # Merge layers take in lists of layers to merge as input.
                # The output of l_sum will be of shape (n_batch, max_n_time_steps, n_features)
                net['l2_lstm'].append(lasagne.layers.ElemwiseSumLayer([net['l2_lstm'][-2], net['l2_lstm'][-1]]))


        net['l3_reshape'] = lasagne.layers.ReshapeLayer(net['l2_lstm'][-1], (-1, n_hidden_list[-1]))

        if debug:
            # Forwards LSTM
            get_l_lstm = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                         L.get_output(net['l2_lstm'][-1]))
            l_lstm_val = get_l_lstm(self.X, self.masks)
            logger_RNNtools.debug('  l2_lstm size: %s', l_lstm_val.shape);

        # Now we can apply feed-forward layers as usual for classification
        net['l6_dense'] = L.DenseLayer(net['l3_reshape'], num_units=num_output_units,
                                       nonlinearity=lasagne.nonlinearities.softmax)

        # Now, the shape will be (n_batch * n_timesteps, num_output_units. We can then reshape to
        # n_batch to get num_output_units values for each timestep from each sequence
        net['l7_out'] = lasagne.layers.ReshapeLayer(net['l6_dense'], (-1, num_output_units))

        if debug:
            get_l_reshape = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                            L.get_output(net['l3_reshape']))
            l_reshape_val = get_l_reshape(self.X, self.masks)
            logger.debug('  l_reshape size: %s', l_reshape_val.shape)

        if debug:   self.print_network_structure(net)

        self.network = net

    def print_network_structure(self, net=None, logger=logger_RNNtools):
        if net==None: net = self.network

        logger.debug("\n PRINTING Network structure: \n %s ", sorted(net.keys()))
        for key in sorted(net.keys()):
            if key == 'l2_lstm':
                for layer in net['l2_lstm']:
                    try:
                        logger.debug('Layer: %12s | in: %s | out: %s', key, layer.input_shape, layer.output_shape)
                    except:
                        logger.debug('Layer: %12s | out: %s', key, layer.output_shape)
            else:
                try:
                    logger.debug('Layer: %12s | in: %s | out: %s', key, net[key].input_shape, net[key].output_shape)
                except:
                    logger.debug('Layer: %12s | out: %s', key, net[key].output_shape)
        return 0

    def use_best_param(self):
        lasagne.layers.set_all_param_values(self.network, self.best_param)
        self.curr_epoch = self.best_epoch
        # Remove the network_train_info enries newer than self.best_epoch
        del self.network_train_info[0][self.best_epoch:]
        del self.network_train_info[1][self.best_epoch:]
        del self.network_train_info[2][self.best_epoch:]

    def load_model(self, model_name, logger=logger_RNNtools):
        if self.network is not None:
            try:
                logger.info("Loading stored model...")

                # restore network weights
                with np.load(model_name) as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                    lasagne.layers.set_all_param_values(self.network['l7_out'], *param_values)

                # # restore 'updates' training parameters
                # with np.load(model_name + "_updates.npz") as f:
                #     updates_values = [f['arr_%d' % i] for i in range(len(f.files))]
                #     for p, value in zip(self.updates.keys(), updates_values):
                #         p.set_value(value)
                logger.info("Loading parameters successful.")
                return 0

            except IOError as e:
                print(os.strerror(e.errno))
                logger.warning('Model: {} not found. No weights loaded'.format(model_name))
                return -1
        else:
            raise IOError('You must build the network before loading the weights.')
        return -1

    def save_model(self, model_name, logger=logger_RNNtools):
        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))
        np.savez(model_name + '.npz', self.best_param)

        # updates_vals = [p.get_value() for p in self.best_updates.keys()]
        # np.savez(model_name + '_updates.npz', updates_vals)

    def create_confusion(self, X, y, debug=False, logger=logger_RNNtools):
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

    def build_functions(self, train=False, debug=False, logger=logger_RNNtools):
        
        # LSTM in lasagne: see https://github.com/craffel/Lasagne-tutorial/blob/master/examples/recurrent.py
        target_var = T.imatrix('targets')

        net_out = self.network['l7_out']
        network_output = L.get_output(net_out)

        # Get the first layer of the network
        l_in = self.network['l1_in']
        l_mask = self.network['l1_mask']

        if debug:  self.print_network_structure()

        # Function to get the output of the network
        #output_fn = theano.function([l_in.input_var, l_mask.input_var], network_output, name='output_fn')
        if debug:
            logger.debug('l_in.input_var.type: \t%s', l_in.input_var.type)
            logger.debug('l_in.input_var.shape:\t%s', l_in.input_var.shape)

            logger.debug('network_output[0]:     \n%s', network_output[0]);
            logger.debug('network_output, shape: \t%s', network_output.shape);


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
        # cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output, target_var.flatten())
        # cost = (cost_pointwise * l_mask.input_var.flatten()).mean()
        cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output, target_var.flatten())
        cost = lasagne.objectives.aggregate(cost_pointwise, l_mask.input_var.flatten())

        # Function to determine the number of correct classifications
        predictions = (T.argmax(network_output, axis=1))
        predictions_fn = theano.function([l_in.input_var, l_mask.input_var], predictions, name='predictions_fn')
        if debug and train:
            predicted = predictions_fn(self.X, self.masks)
            logger.debug('predictions_fn(X).shape: %s', predicted.shape)
            logger.debug('predictions_fn(X)[0], value: \n%s', predicted[0])

        # TODO: only use the output at the middle of each phoneme interval (get better accuracy)
        # Accuracy => # (correctly predicted & valid frames) / #valid frames
        validAndCorrect = T.sum(T.eq(predictions, target_var.flatten()) * l_mask.input_var.flatten())
        nbValidFrames = T.sum(l_mask.input_var.flatten())
        accuracy =  validAndCorrect / nbValidFrames

        # Functions for computing cost and training
        validate_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                      [cost, accuracy], name='validate_fn')
        cost_pointwise_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                            cost_pointwise, name='cost_pointwise_fn')
        if debug and train:
            logger.debug('%s', self.Y.flatten())

            logger.debug('%s', cost_pointwise_fn(self.X, self.masks, self.Y))

            evaluate_cost = validate_fn(self.X, self.masks, self.Y)
            logger.debug('%s %s', type(evaluate_cost), len(evaluate_cost))
            logger.debug('%s', evaluate_cost)
            logger.debug('cost:     {:.3f}'.format(float(evaluate_cost[0])))
            logger.debug('accuracy: {:.3f}'.format(float(evaluate_cost[1])))

        # pdb.set_trace()
        if train:
            LR = T.scalar('LR', dtype=theano.config.floatX)
            # Retrieve all trainable parameters from the network
            all_params = L.get_all_params(net_out, trainable=True)
            self.updates = lasagne.updates.adam(loss_or_grads=cost, params=all_params, learning_rate=LR)
            train_fn = theano.function([l_in.input_var, l_mask.input_var, target_var, LR],
                                       [cost, accuracy], updates=self.updates, name='train_fn')
            self.train_fn = train_fn

        #self.out_fn = output_fn
        self.predictions_fn = predictions_fn
        self.validate_fn = validate_fn

    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1, LR_start=1e-4, LR_decay=1,
              compute_confusion=False, debug=False, logger=logger_RNNtools):

        X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset
        #output_fn = self.out_fn
        predictions_fn = self.predictions_fn
        train_fn = self.train_fn
        validate_fn = self.validate_fn

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

        logger.info("\n* Starting training...")
        LR = LR_start
        for epoch in range(num_epochs):
            self.curr_epoch += 1
            epoch_time = time.time()

            logger.info("CURRENT EPOCH: %s", self.curr_epoch)

            logger.info("Pass over Training Set")
            for inputs, targets, masks, seq_lengths in tqdm(
                    iterate_minibatches(X_train, y_train, valid_frames_train, batch_size, shuffle=False),
                    total=math.ceil(len(X_train) / batch_size)):

                # if debug:
                #     logger.debug('%s %s', inputs.shape, targets.shape)
                #     logger.debug('%s %s', inputs[0].shape, targets[0].shape)

                error, accuracy = train_fn(inputs, masks, targets, LR)
                if debug: logger.debug('%s %s', error, accuracy)
                train_error[epoch] += error
                train_accuracy[epoch] += accuracy
                train_batches[epoch] += 1
                # pdb.set_trace()

            logger.info("Pass over Validation Set")
            for inputs, targets, masks, seq_lengths in iterate_minibatches(X_val, y_val, valid_frames_val, batch_size, shuffle=False):
                error, accuracy = validate_fn(inputs, masks, targets)
                validation_error[epoch] += error
                validation_accuracy[epoch] += accuracy
                validation_batches[epoch] += 1

            logger.info("Pass over Test Set")
            for inputs, targets, masks, seq_lengths in iterate_minibatches(X_test, y_test, valid_frames_test, batch_size, shuffle=False):
                error, accuracy = validate_fn(inputs, masks, targets)
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

            logger.info("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time))

            # better model, so save parameters
            if val_epoch_error < self.best_error:
                self.best_error = val_epoch_error
                self.best_epoch = self.curr_epoch
                self.best_param = lasagne.layers.get_all_param_values(self.network['l7_out'])
                self.best_updates = [p.get_value() for p in self.updates.keys()]
                logger.info("New best model found!")
                if save_name is not None:
                    logger.info("Model saved as " + save_name)
                    self.save_model(save_name)

            logger.info("Learning Rate:\t\t{:.6f} %".format(LR))
            logger.info("Training cost:\t{:.6f}".format(
                    train_error[epoch] / train_batches[epoch]))
            logger.info("  train error:\t\t{:.6f} %".format(train_epoch_error))

            logger.info("Validation cost:\t{:.6f}".format(
                    validation_error[epoch] / validation_batches[epoch]))
            logger.info("  validation error:\t{:.6f} %".format(val_epoch_error))

            logger.info("Test cost:\t\t{:.6f}".format(
                    test_error[epoch] / test_batches[epoch]))
            logger.info("  test error:\t\t{:.6f} %".format(test_epoch_error))

            if compute_confusion:
                confusion_matrices.append(self.create_confusion(X_val, y_val)[0])
                logger.info('  Confusion matrix computed')

            # store train info
            store_path = save_name + '_trainInfo.pkl'
            with open(store_path, 'wb') as cPickle_file:
                cPickle.dump(
                        [self.network_train_info],
                        cPickle_file,
                        protocol=cPickle.HIGHEST_PROTOCOL)
            logger.info("Train info written to:\t %s", store_path)

            if compute_confusion:
                with open(save_name + '_conf.pkl', 'wb') as cPickle_file:
                    cPickle.dump(
                            [confusion_matrices],
                            cPickle_file,
                            protocol=cPickle.HIGHEST_PROTOCOL)

            LR = self.updateLR(LR, LR_decay, logger=logger_RNNtools)

            if self.epochsNotImproved >= 10:
                logging.warning("\n\nNo more improvements, stopping training...")
                break

    def updateLR(self, LR, LR_decay, logger=logger_RNNtools):
        this_error = self.network_train_info[1][-1]
        try:last_error = self.network_train_info[1][-2]
        except: last_error = 10*this_error #first time it will fail because there is only 1 result stored

        # only reduce LR if not much improvment anymore
        if this_error / float(last_error) >= 0.95:
            logger.info(" Error not much reduced: %s vs %s. Reducing LR: %s", this_error, last_error, LR * LR_decay)
            self.epochsNotImproved += 1
            return LR * LR_decay
        else:
            self.epochsNotImproved = max(self.epochsNotImproved - 1, 0)  #reduce by 1, minimum 0
            return LR
                              
