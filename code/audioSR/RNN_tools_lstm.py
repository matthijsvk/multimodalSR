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
import traceback


logger_RNNtools = logging.getLogger('audioSR.tools')
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


    def __init__(self, architecture, dataset=None, batch_size=1, max_seq_length=1000, num_features=26, n_hidden_list=(100,), num_output_units=61,
                 bidirectional=False, addDenseLayers=False, seed=int(time.time()), debug=False, logger=logger_RNNtools):
        self.num_output_units = num_output_units
        self.num_features = num_features
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length #currently unused
        self.epochsNotImproved = 0  #keep track, to know when to stop training
        self.updates = {}
        # self.network_train_info = [[], [], [], [], []]  # train cost, val cost, val acc, test cost, test acc
        self.network_train_info = {
            'train_cost': [],
            'val_cost':   [], 'val_acc': [], 'val_topk_acc': [],
            'test_cost':  [], 'test_acc': [], 'test_topk_acc': []
        }

        if architecture == 'RNN':
            if dataset != None:
                X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset

                X = X_train[:batch_size]
                y = y_train[:batch_size]
                self.valid_frames = valid_frames_train[:batch_size]
                self.masks = generate_masks(X, valid_frames=self.valid_frames, batch_size=len(X))

                self.X = pad_sequences_X(X)
                self.Y = pad_sequences_y(y)
                #self.valid_frames = pad_sequences_y(self.valid_frames)

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

            self.audio_inputs_var = T.tensor3('audio_inputs')
            self.audio_masks_var = T.matrix('audio_masks')  # set MATRIX, not iMatrix!! Otherwise all mask calculations are done by CPU, and everything will be ~2x slowed down!! Also in general_tools.generate_masks()
            self.audio_valid_indices_var = T.imatrix('audio_valid_indices')
            self.audio_targets_var = T.imatrix('audio_targets')

            self.build_RNN(n_hidden_list=n_hidden_list,  bidirectional=bidirectional, addDenseLayers=addDenseLayers,
                           seed=seed, debug=debug, logger=logger)
        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")

    def build_RNN(self, n_hidden_list=(100,), bidirectional=False, addDenseLayers=False,
                  seed=int(time.time()), debug=False, logger=logger_RNNtools):
        # some inspiration from http://colinraffel.com/talks/hammer2015recurrent.pdf

        # if debug:
        #     logger_RNNtools.debug('\nInputs:');
        #     logger_RNNtools.debug('  X.shape:    %s', self.X[0].shape)
        #     logger_RNNtools.debug('  X[0].shape: %s %s %s \n%s', self.X[0][0].shape, type(self.X[0][0]),
        #                           type(self.X[0][0][0]), self.X[0][0][:5])
        #
        #     logger_RNNtools.debug('Targets: ');
        #     logger_RNNtools.debug('  Y.shape:    %s', self.Y.shape)
        #     logger_RNNtools.debug('  Y[0].shape: %s %s %s \n%s', self.Y[0].shape, type(self.Y[0]), type(self.Y[0][0]),
        #                           self.Y[0][:5])
        #     logger_RNNtools.debug('Layers: ')

        # fix these at initialization because it allows for compiler opimizations
        num_output_units = self.num_output_units
        num_features = self.num_features
        batch_size = self.batch_size

        audio_inputs = self.audio_inputs_var
        audio_masks = self.audio_masks_var       #set MATRIX, not iMatrix!! Otherwise all mask calculations are done by CPU, and everything will be ~2x slowed down!! Also in general_tools.generate_masks()
        valid_indices = self.audio_valid_indices_var
        
        
        net = {}
        # net['l1_in_valid'] = L.InputLayer(shape=(batch_size, None), input_var=valid_indices)

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
        gate_parameters = L.recurrent.Gate(
                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                b=lasagne.init.Constant(0.))
        cell_parameters = L.recurrent.Gate(
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

            nextForwardLSTMLayer = L.recurrent.LSTMLayer(
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
                nextBackwardLSTMLayer = L.recurrent.LSTMLayer(
                        input, n_hidden, ingate=gate_parameters,
                        mask_input=net['l1_mask'], forgetgate=gate_parameters,
                        cell=cell_parameters, outgate=gate_parameters,
                        learn_init=True, grad_clipping=100., backwards=True)
                net['l2_lstm'].append(nextBackwardLSTMLayer)

                # if debug:
                #     # Backwards LSTM
                #     get_l_lstm_back = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                #                                       L.get_output(net['l2_lstm'][-1]))
                #     l_lstmBack_val = get_l_lstm_back(self.X, self.masks)
                #     logger_RNNtools.debug('  l_lstm_back size: %s', l_lstmBack_val.shape)

                # We'll combine the forward and backward layer output by summing.
                # Merge layers take in lists of layers to merge as input.
                # The output of l_sum will be of shape (n_batch, max_n_time_steps, n_features)
                net['l2_lstm'].append(L.ElemwiseSumLayer([net['l2_lstm'][-2], net['l2_lstm'][-1]]))

        # we need to convert (batch_size, seq_length, num_features) to (batch_size * seq_length, num_features) because Dense networks can't deal with 2 unknown sizes
        net['l3_reshape'] = L.ReshapeLayer(net['l2_lstm'][-1], (-1, n_hidden_list[-1]))

        # if debug:
        #     get_l_reshape = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
        #                                     L.get_output(net['l3_reshape']))
        #     l_reshape_val = get_l_reshape(self.X, self.masks)
        #     logger.debug('  l_reshape size: %s', l_reshape_val.shape)
        #
        # if debug:
        #     # Forwards LSTM
        #     get_l_lstm = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
        #                                  L.get_output(net['l2_lstm'][-1]))
        #     l_lstm_val = get_l_lstm(self.X, self.masks)
        #     logger_RNNtools.debug('  l2_lstm size: %s', l_lstm_val.shape);

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
        net['l7_out_flattened'] = L.ReshapeLayer(net['l6_dense'], (-1, num_output_units))
        net['l7_out'] = L.ReshapeLayer(net['l6_dense'], (batch_size, -1, num_output_units))
        
        net['l7_out_valid_basic'] = L.SliceLayer(net['l7_out'], indices=valid_indices, axis=1)
        net['l7_out_valid'] = L.ReshapeLayer(net['l7_out_valid_basic'],(batch_size, -1, num_output_units))
        net['l7_out_valid_flattened'] = L.ReshapeLayer(net['l7_out_valid_basic'], (-1, num_output_units))

        if debug:
            get_l_out = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var], L.get_output(net['l7_out']))
            l_out = get_l_out(self.X, self.masks)

            # this only works for batch_size == 1
            get_l_out_valid = theano.function([audio_inputs, audio_masks, valid_indices],
                                         L.get_output(net['l7_out_valid']))
            try:
                l_out_valid = get_l_out_valid(self.X, self.masks, self.valid_frames)
                logger_RNNtools.debug('\n\n\n  l_out: %s  | l_out_valid: %s', l_out.shape, l_out_valid.shape);
            except: logger_RNNtools.warning("batchsize not 1, get_valid not working")


        if debug:   self.print_network_structure(net)
        self.network_lout = net['l7_out_flattened']
        self.network_lout_batch = net['l7_out']
        self.network_lout_valid = net['l7_out_valid']
        self.network_lout_valid_flattened = net['l7_out_valid_flattened']

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
        L.set_all_param_values(self.network, self.best_param)
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
                    L.set_all_param_values(self.network_lout, *param_values)

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
        target_var = self.audio_targets_var #T.imatrix('audio_targets')

        # if debug:  import pdb; self.print_network_structure()


        network_output = L.get_output(self.network_lout_batch)
        network_output_flattened = L.get_output(self.network_lout)  # (batch_size * batch_max_seq_length, nb_phonemes)

        # compare targets with highest output probability. Take maximum of all probs (3rd axis (index 2) of output: 1=batch_size (input files), 2 = time_seq (frames), 3 = n_features (phonemes)
        # network_output.shape = (len(X), 39) -> (nb_inputs, nb_classes)
        predictions = (T.argmax(network_output, axis=2))
        self.predictions_fn = theano.function([self.audio_inputs_var, self.audio_masks_var], predictions,
                                              name='predictions_fn')

        if debug:
            predicted = self.predictions_fn(self.X, self.masks)
            logger.debug('predictions_fn(X).shape: %s', predicted.shape)
            # logger.debug('predictions_fn(X)[0], value: %s', predicted[0])

        if debug:
            self.output_fn = theano.function([self.audio_inputs_var, self.audio_masks_var], network_output, name='output_fn')
            n_out = self.output_fn(self.X, self.masks)
            logger.debug('network_output.shape: \t%s', n_out.shape);
            # logger.debug('network_output[0]:     \n%s', n_out[0]);

        # # Function to determine the number of correct classifications
        # which video, and which frames in the video
        valid_indices_example, valid_indices_seqNr = self.audio_masks_var.nonzero()
        valid_indices_fn = theano.function([self.audio_masks_var], [valid_indices_example, valid_indices_seqNr], name='valid_indices_fn')

        # this gets a FLATTENED array of all the valid predictions of all examples of this batch (so not one row per example)
        # if you want to get the valid predictions per example, you need to use the valid_frames list (it tells you the number of valid frames per wav, so where to split this valid_predictions array)
        # of course this is trivial for batch_size_audio = 1, as all valid_predictions will belong to the one input wav
        valid_predictions = predictions[valid_indices_example, valid_indices_seqNr]
        valid_targets = target_var[valid_indices_example, valid_indices_seqNr]
        self.valid_targets_fn = theano.function([self.audio_masks_var, target_var], valid_targets, name='valid_targets_fn')
        self.valid_predictions_fn = theano.function([self.audio_inputs_var, self.audio_masks_var], valid_predictions, name='valid_predictions_fn')


        # get valid network output
        valid_network_output = network_output[valid_indices_example, valid_indices_seqNr]
        self.valid_network_output_fn = theano.function([self.audio_inputs_var, self.audio_masks_var],
                                                       valid_network_output)

        # using the lasagne SliceLayer:
        # !!!! only works with batch_size == 1  !!!!

        valid_network_output2 = L.get_output(self.network['l7_out_valid'])
        self.valid_network_fn = theano.function([self.audio_inputs_var, self.audio_masks_var,
                                                 self.audio_valid_indices_var], valid_network_output2)
        valid_network_output_flattened = L.get_output(self.network_lout_valid_flattened)

        valid_predictions2 = T.argmax(valid_network_output2,axis=2)
        self.valid_predictions2_fn = theano.function(
                [self.audio_inputs_var, self.audio_masks_var, self.audio_valid_indices_var],
                valid_predictions2, name='valid_predictions_fn')



        # Functions for computing cost and training
        top1_acc = T.mean(lasagne.objectives.categorical_accuracy(
                valid_network_output_flattened, valid_targets.flatten(), top_k=1))
        self.top1_acc_fn = theano.function(
                [self.audio_inputs_var, self.audio_masks_var, self.audio_valid_indices_var,
                 self.audio_targets_var], top1_acc)
        top3_acc = T.mean(lasagne.objectives.categorical_accuracy(
                valid_network_output_flattened, valid_targets.flatten(), top_k=3))
        self.top3_acc_fn = theano.function(
                [self.audio_inputs_var, self.audio_masks_var, self.audio_valid_indices_var,
                 self.audio_targets_var], top3_acc)

        # Functions for computing cost and training
        top1_acc = T.mean(lasagne.objectives.categorical_accuracy(valid_network_output, valid_targets, top_k=1))
        self.top1_acc_fn = theano.function(
                [self.audio_inputs_var, self.audio_masks_var, self.audio_targets_var], top1_acc)
        top3_acc = T.mean(lasagne.objectives.categorical_accuracy(valid_network_output, valid_targets, top_k=3))
        self.top3_acc_fn = theano.function(
                [self.audio_inputs_var, self.audio_masks_var, self.audio_targets_var], top3_acc)

        if debug:
            try:
                # only works with batch_size == 1
                # valid_preds2 = self.valid_predictions2_fn(self.X, self.masks, self.valid_frames)
                # logger.debug("all valid predictions of this batch: ")
                # logger.debug('valid_preds2.shape: %s', valid_preds2.shape)
                # logger.debug('valid_preds2, value: \n%s', valid_preds2)

                # valid_out = self.valid_network_fn(self.X, self.masks, self.valid_frames)
                # logger.debug('valid_out.shape: %s', valid_out.shape)
                # # logger.debug('valid_out, value: \n%s', valid_out)

                valid_example, valid_seqNr = valid_indices_fn(self.masks)
                logger.debug('valid_inds(masks).shape: %s', valid_example.shape)

                valid_output = self.valid_network_output_fn(self.X, self.masks)
                logger.debug("all valid outputs of this batch: ")
                logger.debug('valid_output.shape: %s', valid_output.shape)

                valid_preds = self.valid_predictions_fn(self.X, self.masks)
                logger.debug("all valid predictions of this batch: ")
                logger.debug('valid_preds.shape: %s', valid_preds.shape)
                logger.debug('valid_preds, value: \n%s', valid_preds)

                valid_targs = self.valid_targets_fn(self.masks, self.Y)
                logger.debug('valid_targets.shape: %s', valid_targs.shape)
                logger.debug('valid_targets, value: \n%s', valid_targs)

                top1 = self.top1_acc_fn(self.X, self.masks, self.Y)
                logger.debug("top 1 accuracy: %s", top1*100.0)

                top3 = self.top3_acc_fn(self.X, self.masks, self.Y)
                logger.debug("top 3 accuracy: %s", top3*100.0)

            except Exception as error:
                print('caught this error: ' + traceback.format_exc());
                import pdb; pdb.set_trace()
            #pdb.set_trace()


        ## from https://groups.google.com/forum/#!topic/lasagne-users/os0j3f_Th5Q
        # Pad your vector of labels and then mask the cost:
        # It's important to pad the label vectors with something valid such as zeros,
        # since they will still have to give valid costs that can be multiplied by the mask.
        # The shape of predictions, targets and mask should match:
        # (predictions as (batch_size*max_seq_len, n_features), the other two as (batch_size*max_seq_len,)) -> we need to get the flattened output of the network for this


        # this works, using theano masks
        cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output_flattened, target_var.flatten())
        cost = lasagne.objectives.aggregate(cost_pointwise, self.audio_masks_var.flatten())


        self.validate_fn = theano.function([self.audio_inputs_var, self.audio_masks_var,
                                            self.audio_targets_var],
                                      [cost, top1_acc, top3_acc], name='validate_fn')
        self.cost_pointwise_fn = theano.function([self.audio_inputs_var, self.audio_masks_var, target_var],
                                            cost_pointwise, name='cost_pointwise_fn')

        # # Top k accuracy
        # k = 3
        # topk_acc = T.mean( T.any(T.eq(T.argsort(valid_network_output2, axis=2)[:, -k:],
        #             self.audio_targets_var.dimshuffle(0, 'x')), axis=1),  dtype=theano.config.floatX)
        # topk_acc_fn = theano.function([self.audio_inputs_var, self.audio_masks_var,
        #                                self.audio_valid_indices_var, self.audio_targets_var], topk_acc)

        if debug:
            logger.debug('cost pointwise: %s', self.cost_pointwise_fn(self.X, self.masks, self.Y))

            try:evaluate_cost = self.validate_fn(self.X, self.masks, self.Y)
            except:
                print('caught this error: ' + traceback.format_exc()); pdb.set_trace()
            logger.debug('cost:     {:.3f}'.format(float(evaluate_cost[0])))
            logger.debug('accuracy: {:.3f}'.format(float(evaluate_cost[1]*100.0)))
            logger.debug('top 3 accuracy: {:.3f}'.format(float(evaluate_cost[2]*100.0)))

            #pdb.set_trace()

        if train:
            LR = T.scalar('LR', dtype=theano.config.floatX)
            # Retrieve all trainable parameters from the network
            all_params = L.get_all_params(self.network_lout, trainable=True)
            self.updates = lasagne.updates.adam(loss_or_grads=cost, params=all_params, learning_rate=LR)
            self.train_fn = theano.function([self.audio_inputs_var, self.audio_masks_var,
                                             target_var, LR],
                                       [cost, top1_acc, top3_acc], updates=self.updates, name='train_fn')

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

        cost = 0; accuracy = 0; top3_accuracy= 0
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
            #batch_valid_frames = pad_sequences_y(batch_valid_frames)
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            #import pdb;pdb.set_trace()
            if LR != None: cst, acc, top3_acc = self.train_fn(batch_X, batch_masks, batch_y,  LR)  # training
            else:          cst, acc, top3_acc = self.validate_fn(batch_X, batch_masks, batch_y)   # validation
            cost += cst; accuracy += acc; top3_accuracy += top3_acc

            if get_predictions:
                prediction = self.predictions_fn(batch_X, batch_masks)
                # prediction = np.reshape(prediction, (nb_inputs, -1))  #only needed if predictions_fn is the flattened and not the batched version (see RNN_tools_lstm.py)
                prediction = list(prediction)
                predictions = predictions + prediction
            # # some tests of valid predictions functions (this works :) )
                valid_predictions = self.valid_predictions_fn(batch_X, batch_masks)
                logger_RNNtools.debug("valid predictions: ", valid_predictions.shape)

                # # get valid predictions for video 0
                # self.get_validPredictions_video(valid_predictions, valid_frames, 0)
                # # and the targets for video 0
                # targets[0][valid_frames[0]]
                #
        cost /= nb_batches; accuracy /= nb_batches; top3_accuracy /= nb_batches
        if get_predictions:
            return cost, accuracy*100.0, top3_accuracy*100.0, predictions
        return cost, accuracy*100.0, top3_accuracy*100.0


    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1, LR_start=1e-4, LR_decay=1,
              compute_confusion=False, justTest=False, debug=False, logger=logger_RNNtools):

        X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = dataset

        confusion_matrices = []



        # try to load performance metrics of stored model
        best_val_acc = 0
        test_topk_acc = 0
        test_cost = 0
        test_acc = 0
        try:
            if os.path.exists(save_name + ".npz") and os.path.exists(save_name + "_trainInfo.pkl"):
                old_train_info = unpickle(save_name + '_trainInfo.pkl')
                # backward compatibility
                if type(old_train_info) == list:
                    old_train_info = old_train_info[0]
                    best_val_acc = min(old_train_info[2])
                    test_cost = min(old_train_info[3])
                    test_acc = min(old_train_info[3])
                elif type(old_train_info) == dict:  # normal case
                    best_val_acc = min(old_train_info['val_acc'])
                    test_cost = min(old_train_info['test_cost'])
                    test_acc = min(old_train_info['test_acc'])
                    try:
                        test_topk_acc = min(old_train_info['test_topk_acc'])
                    except:
                        pass
                else:
                    logger.warning("old trainInfo found, but wrong format: %s", save_name + "_trainInfo.pkl")
                    # do nothing
        except:
            pass

        logger.info("Initial best Val acc: %s", best_val_acc)
        logger.info("Initial best test acc: %s\n", test_acc)

        logger.info("Pass over Test Set")
        test_cost, test_accuracy, test_top3_accuracy = self.run_epoch(X=X_test, y=y_test,
                                                                      valid_frames=valid_frames_test)
        logger.info("Test cost:\t\t{:.6f} ".format(test_cost))
        logger.info("Test accuracy:\t\t{:.6f} %".format(test_accuracy))
        logger.info("Test Top 3 accuracy:\t{:.6f} %".format(test_top3_accuracy))

        if justTest: return 0

        logger.info("\n* Starting training...")
        LR = LR_start
        self.best_cost = 100
        for epoch in range(num_epochs):
            self.curr_epoch += 1
            epoch_time = time.time()
            logger.info("CURRENT EPOCH: %s", self.curr_epoch)


            logger.info("Pass over Training Set")
            train_cost, train_accuracy, train_top3_accuracy =     self.run_epoch(X=X_train, y=y_train, valid_frames=valid_frames_train, LR=LR)

            logger.info("Pass over Validation Set")
            validation_cost, validation_accuracy, validation_top3_accuracy =    self.run_epoch(X=X_val, y = y_val, valid_frames=valid_frames_val)


            # Print epoch summary
            logger.info("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time))
            logger.info("Learning Rate:\t\t{:.6f} %".format(LR))
            logger.info("Training cost:\t{:.6f}".format(train_cost))
            logger.info("Validation Top 3 accuracy:\t{:.6f} %".format(validation_top3_accuracy))

            logger.info("Validation cost:\t{:.6f} ".format(validation_cost))
            logger.info("Validation accuracy:\t\t{:.6f} %".format(validation_accuracy))
            logger.info("Validation Top 3 accuracy:\t{:.6f} %".format(validation_top3_accuracy))


            # better model, so save parameters
            if validation_cost < self.best_cost:
                self.best_cost = validation_cost
                self.best_epoch = self.curr_epoch
                self.best_param = L.get_all_param_values(self.network_lout)
                self.best_updates = [p.get_value() for p in self.updates.keys()]
                logger.info("New best model found!")
                if save_name is not None:
                    logger.info("Model saved as " + save_name)
                    self.save_model(save_name)

                logger.info("Pass over Test Set")
                test_cost, test_accuracy, test_top3_accuracy = self.run_epoch(X=X_test, y=y_test,
                                                                              valid_frames=valid_frames_test)
                logger.info("Test cost:\t\t{:.6f} ".format(test_cost))
                logger.info("Test accuracy:\t\t{:.6f} %".format(test_accuracy))
                logger.info("Test Top 3 accuracy:\t{:.6f} %".format(test_top3_accuracy))

            # store train info (old)
            # self.network_train_info[0].append(train_cost)
            # self.network_train_info[1].append(validation_cost)
            # self.network_train_info[2].append(validation_accuracy)
            # self.network_train_info[3].append(test_cost)
            # self.network_train_info[4].append(test_accuracy)

            # save the training info
            self.network_train_info['train_cost'].append(train_cost)
            self.network_train_info['val_cost'].append(validation_cost)
            self.network_train_info['val_acc'].append(validation_accuracy)
            self.network_train_info['val_topk_acc'].append(validation_top3_accuracy)
            self.network_train_info['test_cost'].append(test_cost)
            self.network_train_info['test_acc'].append(test_accuracy)
            self.network_train_info['test_topk_acc'].append(test_top3_accuracy)

            saveToPkl(save_name + '_trainInfo.pkl', self.network_train_info)
            logger.info("Train info written to:\t %s", save_name + '_trainInfo.pkl')

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
        this_cost = self.network_train_info['val_cost'][-1]
        try:last_cost = self.network_train_info['val_cost'][-2]
        except: last_cost = 10*this_cost #first time it will fail because there is only 1 result stored

        # only reduce LR if not much improvment anymore
        if this_cost / float(last_cost) >= 0.98:
            logger.info(" Error not much reduced: %s vs %s. Reducing LR: %s", this_cost, last_cost, LR * LR_decay)
            self.epochsNotImproved += 1
            return LR * LR_decay
        else:
            self.epochsNotImproved = max(self.epochsNotImproved - 1, 0)  #reduce by 1, minimum 0
            return LR