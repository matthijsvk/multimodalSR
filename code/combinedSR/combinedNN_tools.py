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

logger_combinedtools = logging.getLogger('combined.tools')
logger_combinedtools.setLevel(logging.DEBUG)

from general_tools import *
import os
import time
import lasagne
import numpy as np
from preprocessingCombined import *

import pdb


class NeuralNetwork:
    network = None
    training_fn = None
    best_param = None
    best_error = 100
    curr_epoch, best_epoch = 0, 0
    X = None
    Y = None

    network_train_info = [[], [], []]

    def __init__(self, architecture, dataset=None,
                 batch_size=1, num_features=39, num_output_units=39,
                 lstm_hidden_list=(100,), bidirectional=True,
                 cnn_network="google", dense_hidden_list=(512,),
                 seed=int(time.time()), debug=False, logger=logger_combinedtools):

        self.num_output_units = num_output_units
        self.num_features = num_features
        self.batch_size = batch_size
        self.epochsNotImproved = 0  #keep track, to know when to stop training

        if architecture == "combined":
            if dataset != None:
                images_train, mfccs_train, audioLabels_train, validLabels_train, validAudioFrames_train = dataset

                self.images = images_train[0]
                self.mfccs = mfccs_train[0]
                self.audioLabels = audioLabels_train[0]
                self.validLabels = validLabels_train[0]
                self.validAudioFrames = validAudioFrames_train[0]

                self.X = self.mfccs # for debugging audio part
                self.Y = self.audioLabels
                self.valid_frames = self.validAudioFrames

                pdb.set_trace()

                self.masks = generate_masks(self.mfccs, valid_frames=self.validAudioFrames, batch_size=1, logger=logger_combinedtools)
                logger.debug('images.shape:          %s', self.images.shape)
                logger.debug('images[0].shape:       %s', self.images[0].shape)
                logger.debug('images[0][0][0].type:  %s', type(self.images[0][0][0]))
                logger.debug('y.shape:          %s', self.audioLabels.shape)
                logger.debug('y[0].shape:       %s', self.audioLabels[0].shape)
                logger.debug('y[0][0].type:     %s', type(self.audioLabels[0][0]))
                logger.debug('masks.shape:      %s', self.masks.shape)
                logger.debug('masks[0].shape:   %s', self.masks[0].shape)
                logger.debug('masks[0][0].type: %s', type(self.masks[0][0]))

            logger.info("NUM FEATURES: %s", num_features)

            pdb.set_trace()

            # create Theano variables and generate the networks
            RNN_input_var = T.tensor3('audio_inputs')
            RNN_mask_var  = T.matrix('audio_masks')
            RNN_valid_var = T.imatrix('valid_indices')

            self.RNNdict, self.RNN_lout_batch, self.RNN_lout = \
                self.build_RNN(RNN_input_var, RNN_mask_var, RNN_valid_var, n_hidden_list=lstm_hidden_list, bidirectional=bidirectional,
                           seed=seed, debug=debug, logger=logger)

            CNN_input_var = T.tensor4('cnn_input')
            self.CNNdict, self.CNN_lout = self.build_CNN(CNN_input_var)

            self.combined, self.combined_lout = self.build_combined(self.CNN_lout, self.RNN_lout_batch)

        else:
            print("ERROR: Invalid argument: The valid architecture arguments are: 'RNN'")

    def build_RNN(self, RNN_input_var, RNN_mask_var, RNN_valid_var,
                  n_hidden_list=(100,), bidirectional=False,
                  seed=int(time.time()), debug=False, logger=logger_combinedtools):
        # some inspiration from http://colinraffel.com/talks/hammer2015recurrent.pdf

        if debug:
            logger.debug('\nInputs:');
            logger.debug('  X.shape:    %s', self.X[0].shape)
            logger.debug('  X[0].shape: %s %s %s \n%s', self.X[0][0].shape, type(self.X[0][0]),
                                  type(self.X[0][0][0]), self.X[0][0][:5])

            logger.debug('Targets: ');
            logger.debug('  Y.shape:    %s', self.Y.shape)
            logger.debug('  Y[0].shape: %s %s %s \n%s', self.Y[0].shape, type(self.Y[0]), type(self.Y[0][0]),
                                  self.Y[0][:5])
            logger.debug('Layers: ')

        # fix these at initialization because it allows for compiler opimizations
        num_output_units = self.num_output_units
        num_features = self.num_features
        batch_size = self.batch_size

        audio_inputs = RNN_input_var
        audio_masks = RNN_mask_var # set MATRIX, not iMatrix!! Otherwise all mask calculations are done by CPU, and everything will be ~2x slowed down!! Also in general_tools.generate_masks()
        valid_indices = RNN_valid_var

        net = {}

        # shape = (batch_size, batch_max_seq_length, num_features)
        net['l1_in'] = L.InputLayer(shape=(batch_size, None, num_features), input_var=audio_inputs)
        # We could do this and set all input_vars to None, but that is slower -> fix batch_size and num_features at initialization
        # batch_size, n_time_steps, n_features = net['l1_in'].input_var.shape

        # This input will be used to provide the network with masks.
        # Masks are matrices of shape (batch_size, n_time_steps);
        net['l1_mask'] = L.InputLayer(shape=(batch_size, None), input_var=audio_masks)

        if debug:
            get_l_in = L.get_output(net['l1_in'])
            l_in_val = get_l_in.eval({net['l1_in'].input_var: self.X})
            # logger.debug(l_in_val)
            logger.debug('  l_in size: %s', l_in_val.shape);

            get_l_mask = L.get_output(net['l1_mask'])
            l_mask_val = get_l_mask.eval({net['l1_mask'].input_var: self.masks})
            # logger.debug(l_in_val)
            logger.debug('  l_mask size: %s', l_mask_val.shape);

            n_batch, n_time_steps, n_features = net['l1_in'].input_var.shape
            logger.debug("  n_batch: %s | n_time_steps: %s | n_features: %s", n_batch, n_time_steps,
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

            if i == 0:
                input = net['l1_in']
            else:
                input = net['l2_lstm'][i - 1]

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

                if debug:
                    # Backwards LSTM
                    get_l_lstm_back = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var],
                                                      L.get_output(net['l2_lstm'][-1]))
                    l_lstmBack_val = get_l_lstm_back(self.X, self.masks)
                    logger.debug('  l_lstm_back size: %s', l_lstmBack_val.shape)

                # We'll combine the forward and backward layer output by summing.
                # Merge layers take in lists of layers to merge as input.
                # The output of l_sum will be of shape (n_batch, max_n_time_steps, n_features)
                net['l2_lstm'].append(L.ElemwiseSumLayer([net['l2_lstm'][-2], net['l2_lstm'][-1]]))

        # we need to convert (batch_size, seq_length, num_features) to (batch_size * seq_length, num_features) because Dense networks can't deal with 2 unknown sizes
        net['l3_reshape'] = L.ReshapeLayer(net['l2_lstm'][-1], (-1, n_hidden_list[-1]))

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
            logger.debug('  l2_lstm size: %s', l_lstm_val.shape);


        # # Now we can apply feed-forward layers as usual for classification
        net['l6_dense'] = L.DenseLayer(net['l3_reshape'], num_units=num_output_units,
                                       nonlinearity=lasagne.nonlinearities.softmax)

        # # Now, the shape will be (n_batch * n_timesteps, num_output_units). We can then reshape to
        # # n_batch to get num_output_units values for each timestep from each sequence
        # net['l7_out_flattened'] = L.ReshapeLayer(net['l6_dense'], (-1, num_output_units))
        net['l7_out'] = L.ReshapeLayer(net['l6_dense'], (batch_size, -1, num_output_units))

        net['l7_out_valid'] = L.SliceLayer(net['l7_out'], indices=valid_indices, axis=1)
        net['l7_out_valid'] = L.ReshapeLayer(net['l7_out_valid'], (batch_size, -1, num_output_units))

        if debug:
            get_l_out = theano.function([net['l1_in'].input_var, net['l1_mask'].input_var], L.get_output(net['l7_out']))
            l_out = get_l_out(self.X, self.masks)

            get_l_out_valid = theano.function([audio_inputs, audio_masks, valid_indices],
                                              L.get_output(net['l7_out_valid']))
            l_out_valid = get_l_out_valid(self.X, self.masks, self.valid_frames)
            logger.debug('\n\n\n  l_out: %s  | l_out_valid: %s', l_out.shape, l_out_valid.shape);

        if debug:   self.print_RNN_network_structure(net)
        network_output_layer = net['l7_out_flattened']
        network_output_layer_batch = net['l7_out']

        return net, network_output_layer_batch, network_output_layer

    # network from Oxford & Google BBC paper
    def build_CNN(self, input, activation=T.nnet.relu, alpha=0.1, epsilon=1e-4):
        nbClasses = self.num_output_units

        # input
        # store each layer of the network in a dict, for quickly retrieving any layer
        cnnDict = {}
        cnnDict['l0_in'] = lasagne.layers.InputLayer(
                shape=(None, 1, 120, 120),  # 5,120,120 (5 = #frames)
                input_var=input)

        cnnDict['l1_conv1'] = []
        cnnDict['l1_conv1'].append(lasagne.layers.Conv2DLayer(
                cnnDict['l0_in'],
                num_filters=128,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity))
        cnnDict['l1_conv1'].append(lasagne.layers.MaxPool2DLayer(cnnDict['l1_conv1'][-1], pool_size=(2, 2)))
        cnnDict['l1_conv1'].append(lasagne.layers.BatchNormLayer(
                cnnDict['l1_conv1'][-1],
                epsilon=epsilon,
                alpha=alpha))
        cnnDict['l1_conv1'].append(lasagne.layers.NonlinearityLayer(
                cnnDict['l1_conv1'][-1],
                nonlinearity=activation))

        # conv 2
        cnnDict['l2_conv2'] = []
        cnnDict['l2_conv2'].append(lasagne.layers.Conv2DLayer(
                cnnDict['l1_conv1'][-1],
                num_filters=256,
                filter_size=(3, 3),
                stride=(2, 2),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity))
        cnnDict['l2_conv2'].append(lasagne.layers.MaxPool2DLayer(cnnDict['l2_conv2'][-1], pool_size=(2, 2)))
        cnnDict['l2_conv2'].append(lasagne.layers.BatchNormLayer(
                cnnDict['l2_conv2'][-1],
                epsilon=epsilon,
                alpha=alpha))
        cnnDict['l2_conv2'].append(lasagne.layers.NonlinearityLayer(
                cnnDict['l2_conv2'][-1],
                nonlinearity=activation))

        # conv3
        cnnDict['l3_conv3'] = []
        cnnDict['l3_conv3'].append(lasagne.layers.Conv2DLayer(
                cnnDict['l2_conv2'][-1],
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity))
        cnnDict['l3_conv3'].append(lasagne.layers.NonlinearityLayer(
                cnnDict['l3_conv3'][-1],
                nonlinearity=activation))

        # conv 4
        cnnDict['l4_conv4'] = []
        cnnDict['l4_conv4'].append(lasagne.layers.Conv2DLayer(
                cnnDict['l3_conv3'][-1],
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity))
        cnnDict['l4_conv4'].append(lasagne.layers.NonlinearityLayer(
                cnnDict['l4_conv4'][-1],
                nonlinearity=activation))

        # conv 5
        cnnDict['l5_conv5'] = []
        cnnDict['l5_conv5'].append(lasagne.layers.Conv2DLayer(
                cnnDict['l4_conv4'][-1],
                num_filters=512,
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity))
        cnnDict['l5_conv5'].append(lasagne.layers.MaxPool2DLayer(
                cnnDict['l5_conv5'][-1],
                pool_size=(2, 2)))
        cnnDict['l5_conv5'].append(lasagne.layers.NonlinearityLayer(
                cnnDict['l5_conv5'][-1],
                nonlinearity=activation))

        # disable this layer for normal phoneme recognition
        # FC layer
        # cnnDict['l6_fc'] = []
        # cnnDict['l6_fc'].append(lasagne.layers.DenseLayer(
        #         cnnDict['l5_conv5'][-1],
        #        nonlinearity=lasagne.nonlinearities.identity,
        #        num_units=256))
        #
        # cnnDict['l6_fc'].append(lasagne.layers.NonlinearityLayer(
        #         cnnDict['l6_fc'][-1],
        #         nonlinearity=activation))


        # output layer
        cnnDict['l7_out'] = lasagne.layers.DenseLayer(
                # cnnDict['l6_fc'][-1],
                cnnDict['l5_conv5'][-1],
                nonlinearity=lasagne.nonlinearities.softmax,
                num_units=nbClasses)

        # cnn = lasagne.layers.BatchNormLayer(
        #       cnn,
        #       epsilon=epsilon,
        #       alpha=alpha)

        return cnnDict, cnnDict['l7_out']

    def build_combined(self, CNN_lout, RNN_lout, dense_hidden_list):

        # (we process one video at a time)
        # CNN_lout and RNN_lout should be shaped (batch_size, nbFeatures) with batch_size = nb_valid_frames in this video
        # for CNN_lout: nbFeatures = 512x7x7 = 25.088
        # for RNN_lout: nbFeatures = nbUnits(last LSTM layer)
        l_concat = L.ConcatLayer(CNN_lout, RNN_lout)
        l_dense = []
        for i in range(len(dense_hidden_list)):
            n_hidden = dense_hidden_list[i]

            if i == 0:  input = l_concat
            else:       input = l_dense[i - 1]

            nextDenseLayer = L.DenseLayer(input,
                                          nonlinearity=lasagne.nonlinearities.rectify,
                                          num_units=n_hidden)
            # TODO add dropout?
            l_dense.append(nextDenseLayer)
            
        # final softmax layer
        l_out = L.DenseLayer(l_dense[-1], num_units=self.num_output_units,
                                               nonlinearity=lasagne.nonlinearities.softmax)
        return l_out


    def print_RNN_network_structure(self, net=None, logger=logger_combinedtools):
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

    def print_CNN_network_structure(self, net=None, logger=logger_combinedtools):
        if net == None: cnnDict = self.CNNdict
        else: cnnDict = net

        print("\n PRINTING Network structure: \n %s " % (sorted(cnnDict.keys())))
        for key in sorted(cnnDict.keys()):
            print(key)
            if 'conv' in key and type(cnnDict[key]) == list:
                for layer in cnnDict[key]:
                    try:
                        print('      %12s \nin: %s | out: %s' % (layer, layer.input_shape, layer.output_shape))
                    except:
                        print('      %12s \nout: %s' % (layer, layer.output_shape))
            else:
                try:
                    print('Layer: %12s \nin: %s | out: %s' % (
                        cnnDict[key], cnnDict[key].input_shape, cnnDict[key].output_shape))
                except:
                    print('Layer: %12s \nout: %s' % (cnnDict[key], cnnDict[key].output_shape))
        return 0

    def use_best_param(self):
        lasagne.layers.set_all_param_values(self.network, self.best_param)
        self.curr_epoch = self.best_epoch
        # Remove the network_train_info enries newer than self.best_epoch
        del self.network_train_info[0][self.best_epoch:]
        del self.network_train_info[1][self.best_epoch:]
        del self.network_train_info[2][self.best_epoch:]

    def load_model(self, model_type, model_path, logger=logger_combinedtools):
        if self.network is not None:
            try:
                logger.info("Loading stored model...")

                # restore network weights
                with np.load(model_path) as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                    if model_type == 'RNN': lout = self.RNN_lout
                    elif model_type =='CNN': lout = self.CNN_lout
                    elif model_type =='combined': lout = self.combined_lout
                    else:  logger.error('Wrong network type. No weights loaded'.format(model_type))
                    lasagne.layers.set_all_param_values(lout, *param_values)

                logger.info("Loading parameters successful.")
                return 0

            except IOError as e:
                print(os.strerror(e.errno))
                logger.warning('Model: {} not found. No weights loaded'.format(model_path))
                return -1
        else:
            raise IOError('You must build the network before loading the weights.')
        return -1

    def save_model(self, model_name, logger=logger_combinedtools):
        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))
        np.savez(model_name + '.npz', self.best_param)

        # also restore the updates variables to continue training. LR should also be saved and restored...
        # updates_vals = [p.get_value() for p in self.best_updates.keys()]
        # np.savez(model_name + '_updates.npz', updates_vals)

    def create_confusion(self, X, y, debug=False, logger=logger_combinedtools):
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

    def build_functions(self, train=False, debug=False, logger=logger_combinedtools):

        # LSTM in lasagne: see https://github.com/craffel/Lasagne-tutorial/blob/master/examples/recurrent.py
        # and also         http://colinraffel.com/talks/hammer2015recurrent.pdf
        target_var = T.imatrix('targets') # matrix because (example, time_seq)

        # Get the first layer of the network
        l_in = self.network['l1_in']
        l_mask = self.network['l1_mask']

        if debug:
            import pdb; self.print_RNN_network_structure()
            import pdb; self.print_CNN_network_structure()

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


    # Given a dataset and a model, this function trains the model on the dataset for several epochs
    # (There is no default trainer function in Lasagne yet)
    def train(train_fn, val_fn,
              network_output_layer,
              batch_size,
              LR_start, LR_decay,
              num_epochs,
              dataset,
              loadPerSpeaker=False,
              save_name=None,
              shuffleEnabled=True,
              logger=logger_combinedtools):

        if loadPerSpeaker:
            trainingSpeakerFiles, testSpeakerFiles = dataset
            logger.info("train files: \n%s", trainingSpeakerFiles)
            logger.info("test files:  \n %s", testSpeakerFiles)


        # A function which shuffles a dataset
        def shuffle(X, y):
            shuffle_parts = 1
            chunk_size = len(X) / shuffle_parts
            shuffled_range = range(chunk_size)

            X_buffer = np.copy(X[0:chunk_size])
            y_buffer = np.copy(y[0:chunk_size])

            for k in range(shuffle_parts):

                np.random.shuffle(shuffled_range)

                for i in range(chunk_size):
                    X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
                    y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

                X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
                y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

            return X, y

        # This function trains the model a full epoch (on the whole dataset)
        def train_epoch(X, y, LR):
            cost = 0
            LR_np = np.asscalar(np.array([LR]).astype(theano.config.floatX))
            nb_batches = len(X) / batch_size

            for i in tqdm(range(nb_batches), total=nb_batches):
                batch_X = X[i * batch_size:(i + 1) * batch_size]
                batch_y = y[i * batch_size:(i + 1) * batch_size]
                # print("batch_X.shape: ", batch_X.shape)
                # print("batch_y.shape: ", batch_y.shape)
                cost += train_fn(batch_X, batch_y, LR_np)

            return cost, nb_batches

        # This function tests the model a full epoch (on the whole dataset)
        def val_epoch(X, y):
            err = 0
            cost = 0
            nb_batches = len(X) / batch_size

            for i in tqdm(range(nb_batches)):
                batch_X = X[i * batch_size:(i + 1) * batch_size]
                batch_y = y[i * batch_size:(i + 1) * batch_size]
                new_cost, new_err = val_fn(batch_X, batch_y)
                err += new_err
                cost += new_cost

            return err, cost, nb_batches

        # evaluate many TRAINING speaker files -> train loss, val loss and vall error. Load them in one by one (so they fit in memory)
        def evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled, verbose=False):
            train_cost = 0;
            val_err = 0;
            val_cost = 0;
            nb_train_batches = 0;
            nb_val_batches = 0;
            # for each speaker, pass over the train set, then val set. (test is other files). save the results.
            for speakerFile in tqdm(trainingSpeakerFiles, total=len(trainingSpeakerFiles)):
                # TODO: pallelize this with the GPU evaluation to eliminate waiting
                logger.debug("processing %s", speakerFile)
                train, valid, test = getOneSpeaker(
                    speakerFile=speakerFile, trainFraction=0.8, validFraction=0.2, verbose=False)

                # TODO extract images, mfccs, labels, etc from train/valid/test lists

                if verbose:
                    logger.debug("the number of training examples is: %s", len(X_train))
                    logger.debug("the number of valid examples is:    %s", len(X_val))
                    logger.debug("the number of test examples is:     %s", len(X_test))

                if shuffleEnabled: X_train, y_train = shuffle(X_train, y_train)
                train_cost_one, train_batches_one = train_epoch(X=X_train, y=y_train, LR=LR)
                train_cost += train_cost_one;
                nb_train_batches += train_batches_one

                # get results for validation  set
                val_err_one, val_cost_one, val_batches_one = val_epoch(X=X_val, y=y_val)
                val_err += val_err_one;
                val_cost += val_cost_one;
                nb_val_batches += val_batches_one;

                if verbose:
                    logger.debug("  this speaker results: ")
                    logger.debug("\ttraining cost:     %s", train_cost_one / train_batches_one)
                    logger.debug("\tvalidation cost:   %s", val_cost_one / val_batches_one)
                    logger.debug("\vvalidation error rate:  %s %%", val_err_one / val_batches_one)

            # get the average over all speakers
            train_cost /= nb_train_batches
            val_err = val_err / nb_val_batches * 100  # convert to %
            val_cost /= nb_val_batches
            return train_cost, val_cost, val_err

        # evaluate many TEST speaker files. Load them in one by one (so they fit in memory)
        def evalTEST(testSpeakerFiles, verbose=False):
            test_err = 0;
            test_cost = 0;
            nb_test_batches = 0;
            # for each speaker, pass over the train set, then test set. (test is other files). save the results.
            for speakerFile in tqdm(testSpeakerFiles, total=len(testSpeakerFiles)):
                # TODO: pallelize this with the GPU evaluation to eliminate waiting
                logger.debug("processing %s", speakerFile)
                X_train, y_train, X_val, y_val, X_test, y_test = preprocessLipreading.prepLip_one(
                        speakerFile=speakerFile, trainFraction=0.0, validFraction=0.0)

                if verbose:
                    logger.debug("the number of training examples is: %s", len(X_train))
                    logger.debug("the number of valid examples is:    %s", len(X_val))
                    logger.debug("the number of test examples is:     %s", len(X_test))

                # get results for testidation  set
                test_err_one, test_cost_one, test_batches_one = val_epoch(X=X_test, y=y_test)
                test_err += test_err_one;
                test_cost += test_cost_one;
                nb_test_batches += test_batches_one;

                if verbose:
                    logger.debug("  this speaker results: ")
                    logger.debug("\ttest cost:   %s", test_cost_one / test_batches_one)
                    logger.debug("\vtest error rate:  %s %%", test_err_one / test_batches_one)

            # get the average over all speakers
            test_err = test_err / nb_test_batches * 100
            test_cost /= nb_test_batches
            return test_cost, test_err

        def updateLR(LR, LR_decay, network_train_info, epochsNotImproved):
            this_cost = network_train_info[1][-1]  # validation cost
            try:
                last_cost = network_train_info[1][-2]
            except:
                last_cost = 10 * this_cost  # first time it will fail because there is only 1 result stored

            # only reduce LR if not much improvment anymore
            if this_cost / float(last_cost) >= 0.98:
                logger.info(" Error not much reduced: %s vs %s. Reducing LR: %s", this_cost, last_cost,
                                  LR * LR_decay)
                epochsNotImproved += 1
                return LR * LR_decay, epochsNotImproved
            else:
                epochsNotImproved = max(epochsNotImproved - 1, 0)  # reduce by 1, minimum 0
                return LR, epochsNotImproved




        best_val_err = 100
        best_epoch = 1
        LR = LR_start
        # for storage of training info
        network_train_info = [[], [], [], [], []]  # train cost, val cost, val acc, test cost, test acc
        epochsNotImproved = 0

        logger.info("starting training for %s epochs...", num_epochs)
        for epoch in range(num_epochs):
            logger.info("\n\n\n Epoch %s started", epoch + 1)
            start_time = time.time()

            train_cost, val_cost, val_err = evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled)

            # test if validation error went down
            printTest = False
            if val_err <= best_val_err:
                printTest = True
                best_val_err = val_err
                best_epoch = epoch + 1

                logger.info("\n\nBest ever validation score; evaluating TEST set...")

                test_cost, test_err = evalTEST(testSpeakerFiles)

                logger.info("TEST results: ")
                logger.info("\t  test cost:        %s", str(test_cost))
                logger.info("\t  test error rate:  %s %%", str(test_err))

                if save_name is None:
                    save_name = "./bestModel"
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                logger.info("saving model to %s", save_name)
                np.savez(save_name, *lasagne.layers.get_all_param_values(network_output_layer))

            epoch_duration = time.time() - start_time

            # Then we logger.info the results for this epoch:
            logger.info("Epoch %s of %s took %s seconds", epoch + 1, num_epochs, epoch_duration)
            logger.info("  LR:                            %s", str(LR))
            logger.info("  training cost:                 %s", str(train_cost))
            logger.info("  validation cost:               %s", str(val_cost))
            logger.info("  validation error rate:         %s %%", str(val_err))
            logger.info("  best epoch:                    %s", str(best_epoch))
            logger.info("  best validation error rate:    %s %%", str(best_val_err))
            if printTest:
                logger.info("  test cost:                 %s", str(test_cost))
                logger.info("  test error rate:           %s %%", str(test_err))

            # save the training info
            network_train_info[0].append(train_cost)
            network_train_info[1].append(val_cost)
            network_train_info[2].append(val_err)
            network_train_info[3].append(test_cost)
            network_train_info[4].append(test_err)
            store_path = save_name + '_trainInfo.pkl'
            general_tools.saveToPkl(store_path, network_train_info)
            logger.info("Train info written to:\t %s", store_path)

            # decay the LR
            # LR *= LR_decay
            LR = updateLR(LR, LR_decay, network_train_info, epochsNotImproved)

        logger.info("Done.")


    def train(self, dataset, save_name='Best_model', num_epochs=100, batch_size=1, LR_start=1e-4, LR_decay=1,
              compute_confusion=False, debug=False, logger=logger_combinedtools):

        
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
            for inputs, targets, masks, seq_lengths, valid_frames in tqdm(iterate_minibatches(X_train, y_train, valid_frames_train, batch_size, shuffle=True),
                    total=math.ceil(len(X_train) / batch_size)):

                # # some tests of valid predictions functions (this works :) )
                # valid_predictions = self.valid_predictions_fn(inputs, masks)
                # logger.debug("valid predictions: ", valid_predictions.shape)
                #
                # # get valid predictions for video 0
                # self.get_validPredictions_video(valid_predictions, valid_frames, 0)
                # # and the targets for video 0
                # targets[0][valid_frames[0]]

                error, accuracy = train_fn(inputs, masks, targets, LR)
                if debug: logger.debug('%s %s', error, accuracy)
                train_error[epoch] += error
                train_accuracy[epoch] += accuracy
                train_batches[epoch] += 1
                # pdb.set_trace()

            logger.info("Pass over Validation Set")
            for inputs, targets, masks, seq_lengths, valid_frames in tqdm(iterate_minibatches(X_val, y_val, valid_frames_val, batch_size, shuffle=False),total=math.ceil(len(X_val)/batch_size)):
                error, accuracy = validate_fn(inputs, masks, targets)
                validation_error[epoch] += error
                validation_accuracy[epoch] += accuracy
                validation_batches[epoch] += 1

            logger.info("Pass over Test Set")
            for inputs, targets, masks, seq_lengths, valid_frames in tqdm(iterate_minibatches(X_test, y_test, valid_frames_test, batch_size, shuffle=False),total=math.ceil(len(X_test)/batch_size)):
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
            test_epoch_accuracy = test_accuracy[epoch] / test_batches[epoch] * 100

            self.network_train_info[0].append(train_epoch_error)
            self.network_train_info[1].append(val_epoch_error)
            self.network_train_info[2].append(test_epoch_error)

            logger.info("Epoch {} of {} took {:.3f}s.".format(
                    epoch + 1, num_epochs, time.time() - epoch_time))

            # better model, so save parameters
            if val_epoch_error < self.best_error:
                self.best_error = val_epoch_error
                self.best_epoch = self.curr_epoch
                self.best_param = lasagne.layers.get_all_param_values(self.network_output_layer)
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
            logger.info("  test accuracy:\t\t{:.6f} %".format(test_epoch_accuracy))

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

            LR = self.updateLR(LR, LR_decay, logger=logger_combinedtools)

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


    def updateLR(self, LR, LR_decay, logger=logger_combinedtools):
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