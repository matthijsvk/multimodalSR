import numpy as np
import os
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import get_all_param_values
import pdb
# from ctc import CTCLayer
# from TIMIT_utils import create_mask
import pickle
import itertools
# import matplotlib.pyplot as plt 
####### SETTING DEFAULT HYPER-PARAMETER VALUES #######

#Input representation size
INPUT_SIZE = 29

#Hidden layer hyper-parameters
N_HIDDEN = 100
HIDDEN_NONLINEARITY = 'rectify'

#Learning rate
LEARNING_RATE = 0.001

#Number of training sequences (here sentences or examples) in each batch
N_BATCH = 100

#Gradient clipping
GRAD_CLIP = 5

#How often we check the output
EPOCH_SIZE = 100

#Number of epochs to train the system on
NUM_EPOCHS = 100

#Data file name
DATA_FILE_NAME = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CLM.pkl')




# pdb.set_trace()
# def main(num_epochs = NUM_EPOCHS):
print("Building the network...")

l_in = lasagne.layers.InputLayer(shape = (None, None, INPUT_SIZE)) #One-hot represenntation of character indices
l_mask = lasagne.layers.InputLayer(shape = (None, None))

l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_in, num_units=N_HIDDEN, mask_input = l_mask, learn_init=True, grad_clipping=GRAD_CLIP)
Recurrent_output=lasagne.layers.get_output(l_recurrent)

n_batch, n_time_steps, n_features = l_in.input_var.shape

l_reshape = lasagne.layers.ReshapeLayer(l_recurrent, (-1, N_HIDDEN))
Reshape_output = lasagne.layers.get_output(l_reshape)

l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=INPUT_SIZE, nonlinearity = lasagne.nonlinearities.softmax)
# l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_time_steps, n_features))


#Training the network
target_values = T.ivector('target_output') #A vector of character indices linearized over the sequences and batches
target_mask = T.vector('target_mask',dtype=theano.config.floatX)

#Getting the expression for the output (we extract it from the softmax layer to make sure it is compatible with the cross entropy loss function of theano)
network_output = lasagne.layers.get_output(l_dense)

#Calculating the cross-entropy loss
# cost = T.mean(lasagne.objectives.categorical_crossentropy(network_output, target_values))
cost_pointwise = lasagne.objectives.categorical_crossentropy(network_output, target_values)
cost = (cost_pointwise*target_mask).sum()
# cost.reshape([n_batch, n_time_steps, n_features])

all_params = lasagne.layers.get_all_params(l_dense)

updates = lasagne.updates.adam(cost, all_params)



train = theano.function(inputs=[l_in.input_var, target_values, l_mask.input_var, target_mask],	outputs=[cost],	updates=updates)

compute_cost = theano.function(	inputs=[l_in.input_var, target_values, l_mask.input_var, target_mask], 	outputs=[cost], )
# pdb.set_trace()

#################################################################################
#Load data
with open(DATA_FILE_NAME,'rb') as f:
	data = pickle.load(f)
x = data['x']
mask = data['mask']
y = data['y']
y_merged = y.reshape([-1])
mask_merged = mask.reshape([-1])
#Preparing one single batch of inputs

print('Input shape: {}'
	'Total output len: {}'.format(x.shape, len(y_merged)))

Recurrent_output_value=Recurrent_output.eval({l_in.input_var: x, l_mask.input_var: mask})
network_output_value=network_output.eval({l_in.input_var: x, l_mask.input_var: mask}) 
cost_values_pointwise = cost_pointwise.eval({l_in.input_var: x, l_mask.input_var: mask, target_values: y_merged})
cost_value = cost.eval({l_in.input_var: x, l_mask.input_var: mask, target_values: y_merged, target_mask: mask_merged})
print("Recurrent_output"
	"Shape: {}"
	"datatype: {}".format(Recurrent_output_value.shape, type(Recurrent_output_value)))

print("network output"
	"Shape: {}"
	"datatype: {}".format(network_output_value.shape, type(network_output_value)))

print("Pointwise Cost: {}"
	"Cost: {}".format(cost_values_pointwise.shape, cost_value))

cost_vector = []
for epoch in range(NUM_EPOCHS):	
	#pdb.set_trace()
	shuffle_order = np.random.permutation(x.shape[0])
	x = x[shuffle_order, :]
	y = y[shuffle_order, :]
	y_merged = y.reshape([-1])
	mask = mask[shuffle_order, :]
	mask_merged = mask.reshape([-1])


	cost = train(x,y_merged,mask,mask_merged)
	print("Epoch: {}"
		"\tcost = {}".format(epoch,cost) )

	cost_vector.append(cost)
	if epoch % 10 == 0:
		np.savez('CLM_model.npz', *get_all_param_values(l_dense, trainable=True))

# plt.plot(np.arange(NUM_EPOCHS),cost_vector)
# plt.show()
