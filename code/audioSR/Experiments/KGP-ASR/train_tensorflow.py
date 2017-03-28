
# coding: utf-8

# Imports
# ======

# In[1]:

import pickle as pkl
import numpy as np
from time import time
import sys
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import ctc_ops as ctc

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)
  return tf.SparseTensor(x_ix, x_val, x_shape)



def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))
# Load data
# =========

# In[2]:

#f = open('TIMIT_data_prepared_for_CTC.pkl','rb')
f = open('/home/daivik/KGP-ASR/TIMIT_data_prepared_for_CTC.pkl','rb')
data = pkl.load(f)
inp = data['x']
inp1 = data['inputs']
msk = data['mask']
tgt = data['y_indices']
char = data['chars']


# Build the network
# =================


input_size = len(inp1[0][0])
hidden_size = 300
num_output_classes = len(char)
learning_rate = 0.001
output_size = num_output_classes+1
batch_size = None
input_seq_length = None
gradient_clipping = 5
n_time_steps = 776
learningRate = 0.001
momentum = 0.9

# Introduce the targets
# =====================

# In[5]:

# Define the Bi-RNN architecture
# ==============================

# In[6]:

seqLengths = tf.placeholder(tf.int32, shape=(batch_size))
l_in = tf.placeholder(tf.float32, shape=(batch_size, n_time_steps, input_size))
n_batch, n_steps, in_size = l_in.get_shape()
targ_ids = tf.placeholder(tf.int64)
targ_vals = tf.placeholder(tf.int32)
targ_shape = tf.placeholder(tf.int64)
targets = tf.SparseTensor(targ_ids,targ_vals, targ_shape)
l_reshape1 = tf.reshape(l_in,(-1,input_size) )
W1 = weight_variable([input_size,hidden_size])
b1 = bias_variable([hidden_size])
h_1 = tf.nn.relu(tf.matmul(l_reshape1,W1) + b1)
l_reshape2 = tf.reshape(h_1, [-1, n_time_steps,hidden_size])
rnn_input_tr = tf.transpose(l_reshape2,[1,0,2])
rnn_input_re = tf.reshape(rnn_input_tr,[-1,hidden_size])
rnn_input = tf.split(0, n_time_steps, rnn_input_re)
lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    # Get lstm cell output
lstm_outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input, dtype=tf.float32)
lstm_outputs_re=tf.reshape(lstm_outputs, [n_time_steps, -1, 2*hidden_size])
lstm_output_tr=tf.transpose(lstm_outputs_re, [1,0,2])
W2 = weight_variable([2*hidden_size,output_size])
b2 = bias_variable([output_size])
#n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
l_reshape3 = tf.reshape(lstm_output_tr,[-1,2*hidden_size] )
h_2 = tf.matmul(l_reshape3,W2) + b2

l_reshape4 = tf.reshape(h_2,[-1,output_size])

l_soft = tf.nn.softmax(l_reshape4)
l_soft_reshaped = tf.reshape(l_soft,[-1,n_time_steps,output_size])
l_soft_tr = tf.transpose(l_soft_reshaped, [1,0,2])
loss = tf.reduce_mean(tf.nn.ctc_loss(l_soft_tr, targets,seqLengths))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
logitsMaxTest = tf.slice(tf.argmax(l_soft_reshaped, 2), [0, 0], [seqLengths[0], 1])
predictions = tf.to_int32(ctc.ctc_beam_search_decoder(l_soft_reshaped , seqLengths)[0][0])
errorRate = tf.reduce_sum(tf.edit_distance(predictions, targets, normalize=False)) / \
                tf.to_float(tf.size(targets.values))
def getminibatch(x,y,bs):
    perm = np.random.permutation(len(x))
    toselect = perm[:bs]
    batch = {}
    batch['x'] = np.array([x[i] for i in toselect])
    batch['ind'], batch['val'], batch['shape'] = target_list_to_sparse_tensor([y[i] for i in toselect])
    batch['seqlen'] = np.zeros([bs])
    batch['seqlen'].fill(776)
    return batch

number_of_batches = 100
batch_size_var = 38
nEpochs = 100

with tf.Session() as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(number_of_batches)
        #batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        # for batch, batchOrigI in enumerate(batchRandIxs):
        #     batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
        #     batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
        #     feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
        #                 targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        #     _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            # print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            # if (batch % 1) == 0:
            #     print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
            #     print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            # batchErrors[batch] = er*len(batchSeqLengths)
        for i in range(number_of_batches):
            batch = getminibatch(inp,tgt,batch_size_var)
            feedDict = {l_in: batch['x'], targ_ids: batch['ind'], targ_vals: batch['val'],
                        targ_shape: batch['shape'], seqLengths: batch['seqlen']}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            #print(_.get_shape())
            print(np.unique(lmt)) 
            print('Minibatch', i, 'loss:', l)
            print('Minibatch', i,  'error rate:', er)
            batchErrors[i] = er*len(batch['seqlen'])
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
# l_reshape2 = ReshapeLayer(h_1,(n_batch,n_time_steps,hidden_size) )
# l_rec_forward = RecurrentLayer(l_reshape2, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu)
# l_rec_backward = RecurrentLayer(l_reshape2, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu, backwards=True)
# l_rec_accumulation = ElemwiseSumLayer([l_rec_forward,l_rec_backward])
# l_rec_reshaped = ReshapeLayer(l_rec_accumulation, (-1,hidden_size))
# #l_h2 = DenseLayer(l_rec_reshaped, num_units=hidden_size, nonlinearity=clipped_relu)
# l_out = DenseLayer(l_rec_reshaped, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
# l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
# l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
# l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


# # Get the outputs
# # ===============

# # In[7]:

# output_logits = get_output(l_out_reshaped)
# output_softmax = get_output(l_out_softmax_reshaped)


# # Collect all the parameters
# # ==========================

# # In[8]:

# all_params = get_all_params(l_out,trainable=True)
# # print all_params==[l_rec.W_in_to_hid, l_rec.b, l_rec.W_hid_to_hid, l_out.W, l_out.b]


# # In[9]:

# print 'Number of trainable parameters =', len(all_params)
# print all_params==[l_rec_forward.W_in_to_hid, l_rec_forward.b, l_rec_forward.W_hid_to_hid, l_rec_backward.W_in_to_hid, l_rec_backward.b, l_rec_backward.W_hid_to_hid, l_out.W, l_out.b]


# # Compute cost
# # ============

# # In[10]:

# pseudo_cost = ctc_cost.pseudo_cost(y, output_logits)


# # Compute gradients
# # =================

# # In[11]:

# pseudo_cost_grad = T.grad(pseudo_cost.sum() / n_batch, all_params)


# # Compute cost for evaluation
# # ===========================

# # In[12]:

# true_cost = ctc_cost.cost(y, output_softmax)
# cost = T.mean(true_cost)


# # Calculate parameter updates
# # ===========================

# # In[14]:

# shared_learning_rate = theano.shared(lasagne.utils.floatX(0.01))
# updates = lasagne.updates.rmsprop(pseudo_cost_grad, all_params, learning_rate=learning_rate)


# # Define the training op
# # ======================

# # In[15]:

# theano.config.exception_verbosity='high'
# train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)


# # Sanity check the input data
# # ===========================

# # In[16]:

# inp0 = inp1[0]
# inp00= np.asarray([inp0],dtype=theano.config.floatX)
# tgt0 = np.asarray(tgt[0],dtype=np.int16)
# tgt00 = np.asarray([tgt0])
# print inp00.shape, tgt00.shape


# Run Training
# # ============

# # In[19]:

# num_epochs = 100
# #num_training_samples = len(inp1)
# num_training_samples = 3000
# for epoch in range(num_epochs):
#     t = time()
#     cost = 0
#     failures = []

# ##### Step decay of learning rate
#     if(epoch % 30 == 29 ):
# 	shared_learning_rate.set_value(shared_learning_rate.get_value() * 0.1 )
    
#     for i in range(num_training_samples):
#         curr_inp = inp1[i]
# #         curr_msk = msk[i].astype(np.bool)
# #         curr_inp = curr_inp[curr_msk]
#         curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
#         curr_tgt = np.asarray(tgt[i],dtype=np.int16)
#         curr_tgt = np.asarray([curr_tgt])
#         try:
#             _,_,c,_=train(curr_inp,curr_tgt)
#             cost += c
#         except IndexError:
#             failures.append(i)
#             print 'Current input seq: ', curr_inp
#             print 'Current output seq: ', curr_tgt
#             sys.exit(IndexError)
#     f = open('result_3000samples_300param_new_arch','a')
#     f.write('Epoch: '+ str(epoch) +'Cost: '+ str(float(cost/(num_training_samples-len(failures))))+ ', time taken ='+str( time()-t) +'\n')
#     f.close()

#     print 'Epoch: ', epoch, 'Cost: ', float(cost/(num_training_samples-len(failures))), ', time taken =', time()-t
# #     print 'Exceptions: ', len(failures), 'Total examples: ', num_training_samples
#     if epoch%10==0:        
#         #Save the model
#         np.savez('CTC_model_under_test_3000s_300p_new_arch.npz', *get_all_param_values(l_out_softmax_reshaped, trainable=True))
#         for i in range(2):
#             curr_inp = inp1[i]
#             curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
#             curr_tgt = np.asarray(tgt[i],dtype=np.int16)
#             curr_out = output_softmax.eval({l_in.input_var:curr_inp})
#             print 'Predicted:', index2char_TIMIT(np.argmax(curr_out, axis=2)[0])
#             print 'Target:', index2char_TIMIT(curr_tgt)


# In[20]:




# In[21]:



