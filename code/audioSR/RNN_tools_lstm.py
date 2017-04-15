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


class NeuralNetwork:
    network = None
    training_fn = None
    best_param = None
    best_error = 100
    curr_epoch, best_epoch = 0, 0
    X = None
    Y = None

    network_train_info = [[], [], []]

    def __init__(self, architecture, dataset=None, batch_size=1, max_seq_length=1000, num_features=26, n_hidden_list=(100,), num_output_units=61,
                 bidirectional=False, addDenseLayers=False, seed=int(time.time()), debug=False, logger=logger_RNNtools):
        self.num_output_units = num_output_units
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length #currently unused
        self.epochsNotImproved = 0  #keep track, to know when to stop training
        self.updates = {}

        if architecture == 'RNN':
            if dataset != None:
                X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset

                X = X_train[:batch_size]
                y = y_train[:batch_size]
                self.valid_frames = valid_frames_train[:batch_size]
                self.masks = generate_masks(X, valid_frames=self.valid_frames, batch_size=len(X))

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

            self.build_RNN(n_hidden_list=n_hidden_list,  bidirectional=bidirectional, addDenseLayers=addDenseLayers,
                           seed=seed, debug=debug, logger=logger)
        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")

    def build_RNN(self, n_hidden_list=(100,), bidirectional=False, addDenseLayers=False,
                  seed=int(time.time()), debug=False, logger=logger_RNNtools):
        # some inspiration from http://colinraffel.com/talks/hammer2015recurrent.pdf

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

        # fix these at initialization because it allows for compiler opimizations
        num_output_units = self.num_output_units
        num_features = self.num_features
        batch_size = self.batch_size

        audio_inputs = T.tensor3('audio_inputs')
        audio_masks = T.matrix('audio_masks')       #TODO set MATRIX, not iMatrix!! Otherwise all mask calculations are done by CPU, and everything will be ~2x slowed down!! Also in general_tools.generate_masks()

        net = {}
        # shape = (batch_size, batch_max_seq_length, num_features)
        net['l1_in'] = L.InputLayer(shape=(batch_size, None, num_features),input_var=audio_inputs)
        # We could do this and set all input_vars to None, but that is slower -> fix batch_size and num_features at initialization
        # batch_size, n_time_steps, n_features = net['l1_in'].input_var.shape

        # This input will be used to provide the network with masks.
        # Masks are matrices of shape (batch_size, n_time_steps);
        net['l1_mask'] = L.InputLayer(shape=(batch_size, None), input_var=audio_masks)

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

        # we need to convert (batch_size,seq_length, num_features) to (batch_size * seq_length, num_features) because Dense networks can't deal with 2 unknown sizes
        net['l3_reshape'] = lasagne.layers.ReshapeLayer(net['l2_lstm'][-1], (-1, n_hidden_list[-1]))
        if debug:
            get_l_reshape = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                            L.get_output(net['l3_reshape']))
            l_reshape_val = get_l_reshape(self.X, self.masks)
            logger.debug('  l_reshape size: %s', l_reshape_val.shape)

        if debug:
            # Forwards LSTM
            get_l_lstm = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                         L.get_output(net['l2_lstm'][-1]))
            l_lstm_val = get_l_lstm(self.X, self.masks)
            logger_RNNtools.debug('  l2_lstm size: %s', l_lstm_val.shape);

        if addDenseLayers:
            net['l4_dense'] = L.DenseLayer(net['l3_reshape'], nonlinearity =lasagne.nonlinearities.rectify, num_units=256)
            dropoutLayer = L.DropoutLayer(net['l4_dense'], p=0.3)
            net['l5_dense'] = L.DenseLayer(dropoutLayer, nonlinearity=lasagne.nonlinearities.rectify, num_units=64)
            # Now we can apply feed-forward layers as usual for classification
            net['l6_dense'] = L.DenseLayer(net['l5_dense'], num_units=num_output_units,
                                           nonlinearity=lasagne.nonlinearities.softmax)
        else:
            # Now we can apply feed-forward layers as usual for classification
            net['l6_dense'] = L.DenseLayer(net['l3_reshape'], num_units=num_output_units,
                                           nonlinearity=lasagne.nonlinearities.softmax)

        # # Now, the shape will be (n_batch * n_timesteps, num_output_units). We can then reshape to
        # # n_batch to get num_output_units values for each timestep from each sequence
        net['l7_out_flattened'] = lasagne.layers.ReshapeLayer(net['l6_dense'], (-1, num_output_units))
        net['l7_out'] = lasagne.layers.ReshapeLayer(net['l6_dense'], (batch_size, -1, num_output_units))

        if debug:   self.print_network_structure(net)
        self.network_output_layer = net['l7_out_flattened']
        self.network_output_layer_batch = net['l7_out']
        self.network = net

    def print_network_structure(self, net=None, logger=logger_RNNtools):
        if net==None: net = self.network

        logger.debug("\n PRINTING Network structure: \n %s ", sorted(net.keys()))
        for key in sorted(net.keys()):
            if 'lstm' in key:
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
                    lasagne.layers.set_all_param_values(self.network_output_layer, *param_values)

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

        # also restore the updates variables to continue training. LR should also be saved and restored...
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
        # and also         http://colinraffel.com/talks/hammer2015recurrent.pdf
        target_var = T.imatrix('targets') # matrix because (example, time_seq)

        # Get the first layer of the network
        l_in = self.network['l1_in']
        l_mask = self.network['l1_mask']

        if debug:  import pdb; self.print_network_structure()

        batch = True
        if batch:
            network_output = L.get_output(self.network_output_layer_batch)
            network_output_flattened = L.get_output( self.network_output_layer)  # (batch_size * batch_max_seq_length, nb_phonemes)

            # compare targets with highest output probability. Take maximum of all probs (3rd axis (index 2) of output: 1=batch_size (input files), 2 = time_seq (frames), 3 = n_features (phonemes)
            # network_output.shape = (len(X), 39) -> (nb_inputs, nb_classes)
            predictions = (T.argmax(network_output, axis=2))
            self.predictions_fn = theano.function([l_in.input_var, l_mask.input_var], predictions,
                                                  name='predictions_fn')

            if debug:
                predicted = self.predictions_fn(self.X, self.masks)
                logger.debug('predictions_fn(X).shape: %s', predicted.shape)
                logger.debug('predictions_fn(X)[0], value: %s', predicted[0])

            if debug:
                self.output_fn = theano.function([l_in.input_var, l_mask.input_var], network_output, name='output_fn')
                n_out = self.output_fn(self.X, self.masks)
                logger.debug('network_output[0]:     \n%s', n_out[0]);
                logger.debug('network_output.shape: \t%s', n_out.shape);

            # # Function to determine the number of correct classifications
            valid_indices_example, valid_indices_seqNr = l_mask.input_var.nonzero()
            valid_indices_fn = theano.function([l_mask.input_var], [valid_indices_example, valid_indices_seqNr], name='valid_indices_fn')

            # this gets a FLATTENED array of all the valid predictions of all examples of this batch (so not one row per example)
            # if you want to get the valid predictions per example, you need to use the valid_frames list (it tells you the number of valid frames per wav, so where to split this valid_predictions array)
            # of course this is trivial for batch_size_audio = 1, as all valid_predictions will belong to the one input wav
            valid_predictions = predictions[valid_indices_example, valid_indices_seqNr]
            valid_targets = target_var[valid_indices_example, valid_indices_seqNr]
            self.valid_targets_fn = theano.function([l_mask.input_var, target_var], valid_targets, name='valid_targets_fn')
            self.valid_predictions_fn = theano.function([l_in.input_var, l_mask.input_var], valid_predictions, name='valid_predictions_fn')

            if debug:
                try:
                    valid_example, valid_seqNr = valid_indices_fn(self.masks)
                    logger.debug('valid_inds(masks).shape: %s', valid_example.shape)
                    valid_preds = self.valid_predictions_fn(self.X, self.masks)
                    logger.debug("all valid predictions of this batch: ")
                    logger.debug('valid_preds(X,masks).shape: %s', valid_preds.shape)
                    logger.debug('valid_preds(X,masks)[0], value: \n%s', valid_preds)

                    valid_targs = self.valid_targets_fn(self.masks, self.Y)
                    logger.debug("all valid targets of this batch: ")
                    logger.debug('valid_targets(X,masks).shape: %s', valid_targs.shape)
                    logger.debug('valid_targets(X,masks)[0], value: \n%s', valid_targs)
                    # pdb.set_trace()
                except Exception as error:
                    print('caught this error: ' + traceback.format_exc());
                    pdb.set_trace()

            # only use the output at the middle of each phoneme interval (get better accuracy)
            # Accuracy => # (correctly predicted & valid frames) / #valid frames
            validAndCorrect = T.sum(T.eq(valid_predictions, valid_targets),dtype='float32')
            nbValidFrames = T.sum(l_mask.input_var)
            accuracy =  validAndCorrect / nbValidFrames

        else:
            # Function to get the output of the network
            # network_output = L.get_output(self.network_output_layer_batch)                     # (batch_size, batch_max_seq_length, nb_phonemes)
            network_output_flattened = L.get_output(self.network_output_layer) # (batch_size * batch_max_seq_length, nb_phonemes)

            # valid predictions
            eqs = T.neq(l_mask.input_var.flatten(), T.zeros((1,)))
            valid_indices = eqs.nonzero()[0]
            valid_indices_fn = theano.function([l_mask.input_var], valid_indices, name='valid_indices_fn')
            valid_predictions = network_output_flattened[valid_indices, :]
            self.valid_predictions_fn = theano.function([l_in.input_var, l_mask.input_var], valid_predictions,
                                                        name='valid_predictions_fn')

            # the flattened version; faster because we need flattened stuff anyway when calculating loss.
            # If we used the batched version here, we would need to calculate both batched and flattened predictions, which is double work.
            predictions_flattened = (T.argmax(network_output_flattened, axis=1))
            self.predictions_fn = theano.function([l_in.input_var, l_mask.input_var], predictions_flattened,
                                                  name='predictions_fn')
            validAndCorrect = T.sum(T.eq(predictions_flattened, target_var.flatten()) * l_mask.input_var.flatten())
            nbValidFrames = T.sum(l_mask.input_var.flatten())
            accuracy = validAndCorrect / nbValidFrames


        ## from https://groups.google.com/forum/#!topic/lasagne-users/os0j3f_Th5Q
        # Pad your vector of labels and then mask the cost:
        # It's important to pad the label vectors with something valid such as zeros,
        # since they will still have to give valid costs that can be multiplied by the mask.
        # The shape of predictions, targets and mask should match:
        # (predictions as (batch_size*max_seq_len, n_features), the other two as (batch_size*max_seq_len,)) -> we need to get the flattened output of the network for this
        cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output_flattened, target_var.flatten())
        cost = lasagne.objectives.aggregate(cost_pointwise, l_mask.input_var.flatten())

        # Functions for computing cost and training
        self.validate_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                      [cost, accuracy], name='validate_fn')
        self.cost_pointwise_fn = theano.function([l_in.input_var, l_mask.input_var, target_var],
                                            cost_pointwise, name='cost_pointwise_fn')
        if debug:
            logger.debug('cost pointwise: %s', self.cost_pointwise_fn(self.X, self.masks, self.Y))

            try:evaluate_cost = self.validate_fn(self.X, self.masks, self.Y)
            except:
                print('caught this error: ' + traceback.format_exc()); pdb.set_trace()
            logger.debug('evaluate_cost: %s %s', type(evaluate_cost), len(evaluate_cost))
            logger.debug('%s', evaluate_cost)
            logger.debug('cost:     {:.3f}'.format(float(evaluate_cost[0])))
            logger.debug('accuracy: {:.3f}'.format(float(evaluate_cost[1])))
            #pdb.set_trace()

        if train:
            LR = T.scalar('LR', dtype=theano.config.floatX)
            # Retrieve all trainable parameters from the network
            all_params = L.get_all_params(self.network_output_layer, trainable=True)
            self.updates = lasagne.updates.adam(loss_or_grads=cost, params=all_params, learning_rate=LR)
            self.train_fn = theano.function([l_in.input_var, l_mask.input_var, target_var, LR],
                                       [cost, accuracy], updates=self.updates, name='train_fn')

    def shuffle(X, y, valid_frames):

        chunk_size = len(X)
        shuffled_range = range(chunk_size)

        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        valid_frames_buffer = np.copy(valid_frames[0:chunk_size])

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):
            X_buffer[i] = X[shuffled_range[i]]
            y_buffer[i] = y[shuffled_range[i]]
            valid_frames_buffer[i] = valid_frames[shuffled_range[i]]

        X[0: chunk_size] = X_buffer
        y[0: chunk_size] = y_buffer
        valid_frames[0: chunk_size] = valid_frames_buffer

        return X, y, valid_frames

    # This function trains the model a full epoch (on the whole dataset)
    def run_epoch(self, X, y, valid_frames, get_predictions= False, LR=None, batch_size = -1):
        if batch_size == -1: batch_size= self.batch_size

        cost = 0; accuracy = 0
        nb_batches = len(X) / batch_size

        predictions = [] #only used if get_predictions = True
        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            batch_valid_frames = valid_frames[i * batch_size:(i + 1) * batch_size]
            batch_masks = generate_masks(batch_X, valid_frames=batch_valid_frames, batch_size=batch_size)
            # now pad inputs and target to maxLen
            batch_X = pad_sequences_X(batch_X)
            batch_y = pad_sequences_y(batch_y)
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            if LR != None: cst, acc = self.train_fn(batch_X, batch_masks, batch_y, LR)  # training
            else:          cst, acc = self.validate_fn(batch_X, batch_masks, batch_y)   # validation
            cost += cst; accuracy += acc

            if get_predictions:
                prediction = self.predictions_fn(batch_X, batch_masks)
                # prediction = np.reshape(prediction, (nb_inputs, -1))  #only needed if predictions_fn is the flattened and not the batched version (see RNN_tools_lstm.py)
                prediction = list(prediction)
                predictions = predictions + prediction
            # # some tests of valid predictions functions (this works :) )
            #     # valid_predictions = self.valid_predictions_fn(inputs, masks)
            #     # logger.debug("valid predictions: ", valid_predictions.shape)
            #     #
            #     # # get valid predictions for video 0
            #     # self.get_validPredictions_video(valid_predictions, valid_frames, 0)
            #     # # and the targets for video 0
            #     # targets[0][valid_frames[0]]
            
        cost /= nb_batches; accuracy /= nb_batches
        if get_predictions:
            return cost, accuracy, predictions
        return cost, accuracy


    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1, LR_start=1e-4, LR_decay=1,
              compute_confusion=False, debug=False, logger=logger_RNNtools):

        X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset

        # Initiate some vectors used for tracking performance
        train_cost = np.zeros([num_epochs])
        train_accuracy = np.zeros([num_epochs])

        validation_cost = np.zeros([num_epochs])
        validation_accuracy = np.zeros([num_epochs])

        test_cost = np.zeros([num_epochs])
        test_accuracy = np.zeros([num_epochs])

        confusion_matrices = []

        logger.info("\n* Starting training...")
        LR = LR_start
        for epoch in range(num_epochs):
            self.curr_epoch += 1
            epoch_time = time.time()
            logger.info("CURRENT EPOCH: %s", self.curr_epoch)


            logger.info("Pass over Training Set")
            train_cost[epoch], train_accuracy[epoch] = \
                self.run_epoch(X=X_train, y=y_train, valid_frames=valid_frames_train, LR=LR)

            logger.info("Pass over Validation Set")
            validation_cost[epoch], validation_accuracy[epoch] = \
                self.run_epoch(X=X_val, y = y_val, valid_frames=valid_frames_val)

            logger.info("Pass over Test Set")
            test_cost[epoch], test_accuracy[epoch] =\
                self.run_epoch(X=X_test, y=y_test, valid_frames=valid_frames_test)


            # Print epoch summary
            logger.info("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time))

            # better model, so save parameters
            if validation_cost[epoch] < self.best_cost:
                self.best_cost = validation_cost[epoch]
                self.best_epoch = self.curr_epoch
                self.best_param = lasagne.layers.get_all_param_values(self.network_output_layer)
                self.best_updates = [p.get_value() for p in self.updates.keys()]
                logger.info("New best model found!")
                if save_name is not None:
                    logger.info("Model saved as " + save_name)
                    self.save_model(save_name)

            logger.info("Learning Rate:\t\t{:.6f} %".format(LR))
            logger.info("Training cost:\t{:.6f}".format(train_cost[epoch]))
            logger.info("Validation cost:\t{:.6f} ".format(validation_cost[epoch]))
            logger.info("Test cost:\t\t{:.6f} ".format(test_cost[epoch]))
            logger.info("Test accuracy:\t\t{:.6f} %".format(test_accuracy[epoch]))

            # store train info
            self.network_train_info[0].append(train_cost[epoch])
            self.network_train_info[1].append(validation_cost[epoch])
            self.network_train_info[2].append(test_cost[epoch])
            store_path = save_name + '_trainInfo.pkl'
            with open(store_path, 'wb') as cPickle_file:
                cPickle.dump(
                        [self.network_train_info],
                        cPickle_file,
                        protocol=cPickle.HIGHEST_PROTOCOL)
            logger.info("Train info written to:\t %s", store_path)

            if compute_confusion:
                confusion_matrices.append(self.create_confusion(X_val, y_val)[0])
                logger.info('  Confusion matrix computed')
                with open(save_name + '_conf.pkl', 'wb') as cPickle_file:
                    cPickle.dump(
                            [confusion_matrices],
                            cPickle_file,
                            protocol=cPickle.HIGHEST_PROTOCOL)

            # update LR, see if we can stop training
            LR = self.updateLR(LR, LR_decay, logger=logger_RNNtools)

            if self.epochsNotImproved >= 5:
                logging.warning("\n\nNo more improvements, stopping training...")
                break

    def get_validPredictions_video(self, valid_predictions, valid_frames, videoIndexInBatch):
        # get indices of the valid frames for each video, using the valid_frames
        nbValidsPerVideo = [len(el) for el in valid_frames]

        # each el is the sum of the els before. -> for example video 3, you need valid_predictions from indices[2] (inclusive) till indices[3] (not inclusive)
        indices = [0] + [np.sum(nbValidsPerVideo[:i + 1]) for i in range(len(nbValidsPerVideo))]

        # make a 2D list. Each el of the list is a list with the valid frames per video.
        videoPreds = [range(indices[videoIndex], indices[videoIndex + 1]) for videoIndex in range(
                len(valid_frames))]
        #assert len(videoPreds) == len(inputs) == len(valid_frames)

        # now you can get the frames for a specific video:
        return valid_predictions[videoPreds[videoIndexInBatch]]


    def updateLR(self, LR, LR_decay, logger=logger_RNNtools):
        this_error = self.network_train_info[1][-1]
        try:last_error = self.network_train_info[1][-2]
        except: last_error = 10*this_error #first time it will fail because there is only 1 result stored

        # only reduce LR if not much improvment anymore
        if this_error / float(last_error) >= 0.98:
            logger.info(" Error not much reduced: %s vs %s. Reducing LR: %s", this_error, last_error, LR * LR_decay)
            self.epochsNotImproved += 1
            return LR * LR_decay
        else:
            self.epochsNotImproved = max(self.epochsNotImproved - 1, 0)  #reduce by 1, minimum 0
            return LR