import theano
import theano.tensor as T
import lasagne 
from lasagne.layers import InputLayer, DenseLayer, RecurrentLayer, NonlinearityLayer, ReshapeLayer, get_output, get_all_params, get_all_param_values, ElemwiseSumLayer
import ctc_cost
from time import time
from special_activations import clipped_relu
import sys
from collections import defaultdict
import random
import heapq
import numpy as np
import os
import soundfile as sf
from features import mfcc
import pickle
import sys
import theano
import pdb
from TIMIT_utils import *
'''TODO: Some Hard coded stuff '''

CLM_LEN = 78
def wav_to_input( wav_file_name ):
	input_data, f_s = sf.read(wav_file_name)
	# mfcc_feat = MFCC_input(mfcc(input_data,f_s))
	mfcc_feat = mfcc(input_data,f_s)
	#Delta features
	delta_feat = mfcc_feat[:-1]-mfcc_feat[1:]
	#Delta-Delta features
	deltadelta_feat = delta_feat[:-1]-delta_feat[1:]

	#Removing the first two frames
	mfcc_feat = mfcc_feat[2:]
	delta_feat = delta_feat[1:]

	#Concatenating mfcc, delta and delta-delta features
	full_input = np.concatenate((mfcc_feat,delta_feat,deltadelta_feat), axis=1)

	return full_input

def predictWithCLM( sequence ):
	''' Returns dictionary of character probabilities '''
	#seq2 = sequence.replace("_","")
	if(len(sequence)>78):
		seq_truncated = sequence[-78:]
	else:
		seq_truncated = sequence
	X = getOneHot(seq_truncated)
	mask = np.zeros(78)
	mask[:len(seq_truncated)] = 1
	out_prob = CLM.eval({CLM_in.input_var : [X], CLM_mask.input_var : [mask]})
	dic = {}
	for i in range(len(list_of_alphabets)):
		dic[list_of_alphabets[i]] = out_prob[len(seq_truncated)-1][i];

	return dic

def getOneHot( sequence ):
	''' returns 1 hot representation of the sequence '''
	input_length = len(sequence)
	input_dim = 29
	max_input_length = 78
	X = np.zeros( (max_input_length, input_dim) )
	for i in range(input_length):
		if(sequence[i] == "_"):
			continue
		X[i][list_of_alphabets.index(sequence[i])] = 1
	return X


def decode( input_data ):
	p_b = {}
	p_nb = {}
	p_tot = {}
	alpha = 1.25
	beta = 1.5
	k = 20
	p_ctc=[]
	for i in range(len(input_data)):
		dic = {}
		for j in range(len(list_of_alphabets)):
			dic[list_of_alphabets[j]] = input_data[i][j]
		dic["_"]=input_data[i][len(list_of_alphabets)] # _ is the last output
		p_ctc.append(dic)
	#"_" => Empty String from the psuedocode in the lexfree paper
	p_b["_"] = 1 
	p_nb["_"] = 0
	p_tot["_"] = 1
	z_prev = ["_"]
	print("Number of frames:"+str(len(input_data)))
	for t in range(len(input_data)):
		print(str(t)+"/"+str(len(input_data))+" done")
		z_next = []
		p_b_next={}
		p_nb_next = {}
		for string in z_prev:
			p_clm = predictWithCLM(string)
			p_b_next[string] = p_ctc[t]["_"] * p_tot[string]
			p_nb_next[string] = p_ctc[t][string[-1]] * p_nb[string]
			z_next.append(string)
			for char in list_of_alphabets:
				new_string = string + char
				if(char != string[-1]):
					p_nb_next[new_string] = p_ctc[t][char]*(p_clm[char]**alpha)*p_tot[string]
				else:
					p_nb_next[new_string] = p_ctc[t][char]*(p_clm[char]**alpha)*p_b[string]
				if( new_string not in p_b_next ):
					p_b_next[new_string] = 0
				z_next.append(new_string)
		p_tot = {}
		plen = {}
		for string in z_next:
			p_tot[string] = p_b_next[string] + p_nb_next[string]
			plen[string] = p_tot[string]*(len(string)**beta)
		p_b = p_b_next
		p_nb = p_nb_next
		z_prev = sorted(plen, key=plen.get)[-k:]#get max k keys
	d=[]
	for i in z_prev:
		d.append((p_b[i]+p_nb[i]))
	ind=d.index(max(d))
	ans=z_prev[ind]
	return ans
def getTrainedRNN():
	''' Read from file and set the params (To Do: Refactor 
		so as to do this only once) '''
	input_size = 39
	hidden_size = 50
	num_output_classes = 29
	learning_rate = 0.001
	output_size = num_output_classes+1
	batch_size = None
	input_seq_length = None
	gradient_clipping = 5

	l_in = InputLayer(shape=(batch_size, input_seq_length, input_size))
	n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
	# h_1 = DenseLayer(l_in, num_units=hidden_size, nonlinearity=clipped_relu)
	l_rec_forward = RecurrentLayer(l_in, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu)
	l_rec_backward = RecurrentLayer(l_in, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu, backwards=True)
	l_rec_accumulation = ElemwiseSumLayer([l_rec_forward,l_rec_backward])
	l_rec_reshaped = ReshapeLayer(l_rec_accumulation, (-1,hidden_size))
	l_h2 = DenseLayer(l_rec_reshaped, num_units=hidden_size, nonlinearity=clipped_relu)
	l_out = DenseLayer(l_h2, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
	l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
	l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
	l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


	with np.load('CTC_model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(l_out_softmax_reshaped, param_values, trainable = True)
	output = lasagne.layers.get_output( l_out_softmax_reshaped )
	return l_in, output

def getTrainedCLM():
	''' Read CLM from file '''
	#Some parameters for the CLM
	INPUT_SIZE = 29

	#Hidden layer hyper-parameters
	N_HIDDEN = 100
	HIDDEN_NONLINEARITY = 'rectify'

	#Gradient clipping
	GRAD_CLIP = 100
	l_in = lasagne.layers.InputLayer(shape = (None, None, INPUT_SIZE)) #One-hot represenntation of character indices
	l_mask = lasagne.layers.InputLayer(shape = (None, None))

	l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_in, num_units=N_HIDDEN, mask_input = l_mask, learn_init=True, grad_clipping=GRAD_CLIP)
	Recurrent_output=lasagne.layers.get_output(l_recurrent)

	n_batch, n_time_steps, n_features = l_in.input_var.shape

	l_reshape = lasagne.layers.ReshapeLayer(l_recurrent, (-1, N_HIDDEN))
	Reshape_output = lasagne.layers.get_output(l_reshape)

	l_h1 = lasagne.layers.DenseLayer(l_reshape, num_units=N_HIDDEN)
	l_h2 = lasagne.layers.DenseLayer(l_h1, num_units=N_HIDDEN)
	l_dense = lasagne.layers.DenseLayer(l_h2, num_units=INPUT_SIZE, nonlinearity = lasagne.nonlinearities.softmax)
	with np.load('CLM_model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(l_dense, param_values,trainable = True)
	output = lasagne.layers.get_output( l_dense )
	return l_in,l_mask,output


#def getCLMOneHot( sequence ):
TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CTC.pkl')
with open(TIMIT_pkl_file,'rb') as f:
		data = pickle.load(f)
		list_of_alphabets = data['chars']
RNN_in, BiRNN = getTrainedRNN()
CLM_in, CLM_mask, CLM = getTrainedCLM()
#input_data = wav_to_input('/home/daivik/Downloads/fsew0_v1.1/fsew0_007.wav')
test_sample = int(sys.argv[1])
input_data = data['x'][test_sample];
pred = BiRNN.eval({RNN_in.input_var: [input_data]})
print(decode(pred[0]))
print("Argmax Result:")
print(index2char_TIMIT(np.argmax(pred, axis = 2)[0]))
print(data['y_char'][test_sample])
