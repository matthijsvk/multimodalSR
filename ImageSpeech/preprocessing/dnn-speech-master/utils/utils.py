from random import uniform
import numpy as np
import theano
from theano import config
import theano.tensor as tensor
from theano.tensor.signal import downsample
from collections import OrderedDict, defaultdict
from mlpmodel.mlpClassifier import MlpClassifier
from rnnmodel.rnnClassifier import RnnClassifier
from dbnmodel.DBN import DbnClassifier

def getModelObj(params):
  if params['model_type'] == 'MLP':
    mdl = MlpClassifier(params) 
  elif params['model_type'] == 'DBN':
    mdl = DbnClassifier(params)
  elif params['model_type'] == 'RNN':  
    mdl = RnnClassifier(params) 
  else:
    raise ValueError('ERROR: %s --> This model type is not yet supported'%(params['model_type']))
  return mdl


def randi(N):
  """ get random integer in range [0, N) """
  return int(uniform(0, N))

def merge_init_structs(s0, s1):
  """ merge struct s1 into s0 """
  for k in s1['model']:
    assert (not k in s0['model']), 'Error: looks like parameter %s is trying to be initialized twice!' % (k, )
    s0['model'][k] = s1['model'][k] # copy over the pointer
  s0['update'].extend(s1['update'])
  s0['regularize'].extend(s1['regularize'])

def initw(n,d): # initialize matrix of this size
  magic_number = 0.1
  return (np.random.rand(n,d) * 2 - 1) * magic_number # U[-0.1, 0.1]

def initwTh(n,d,magic_number=0.1): # initialize matrix of this size
  return ((np.random.rand(n,d) * 2 - 1) * magic_number).astype(config.floatX) # U[-0.1, 0.1]

def _p(pp, name):
    return '%s_%s' % (pp, name)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def accumNpDicts(d0, d1):
  """ forall k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
  for k in d1:
    if k in d0:
      d0[k] += d1[k]
    else:
      d0[k] = d1[k]

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    if type(tparams) == list:
        for i in xrange(len(params)):
            tparams[i].set_value(params[i])
    else:
        for kk, vv in params.iteritems():
            tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    if type(zipped) == list:
        new_params = [] 
        for vv in zipped:
            new_params.append(vv.get_value())
    else:
        new_params = OrderedDict()
        for kk, vv in zipped.iteritems():
            new_params[kk] = vv.get_value()
    return new_params

def softmax(x,axis = -1):
    xs = x.shape
    ndim = len(xs)
    if axis == -1:
        axis = ndim -1

    z = np.max(x,axis=axis)
    y = x - z[...,np.newaxis] # for numerical stability shift into good numerical range
    e1 = np.exp(y) 
    p1 = e1 / np.sum(e1,axis=axis)[...,np.newaxis]
    
    return p1

def cosineSim(x,y):
    n1 = np.sqrt(np.sum(x**2)) 
    n2 = np.sqrt(np.sum(y**2)) 
    sim = x.T.dot(y)/(n1*n2) if n1 !=0.0 and n2!= 0.0 else 0.0
    return sim 

def sliceT(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

#Theano functions
def ReLU(x):
    y = tensor.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = tensor.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = tensor.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def myMaxPool(x, ps=[],method='downsamp'):
    if method == 'downsamp':
        y = downsample.max_pool_2d(input= x, ds=ps, ignore_border=True)
    elif method == 'max':
        y = tensor.max(x, axis=3).max(axis=2)
    return(y)

def make_shared(data_set, data_name):
    data_set_th = theano.shared(np.asarray(data_set, dtype=config.floatX), name=data_name, borrow=True)
    return data_set_th

  
def basic_lstm_layer(tparams, state_below, aux_input, use_noise, options, prefix='lstm', sched_prob_mask = []):
  nsteps = state_below.shape[0]
  h_depth = options.get('hidden_depth',1)
  h_sz = options['hidden_size']
  
  if state_below.ndim == 3:
      n_samples = state_below.shape[1]
  else:
      n_samples = 1

  def _step(x_in, xp_m,  h_, c_, xwout_, xAux):
      preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
      if options.get('sched_sampling_mode',None) == None:
        preact += x_in
      else:
        xy_emb = tensor.dot(xwout_, tparams[_p(prefix, 'W_inp')] + tparams[_p(prefix, 'b')])
        temp_container = tensor.concatenate([xy_emb.dimshuffle('x',0,1), x_in.dimshuffle('x', 0, 1)],axis=0)
        preact += temp_container[ xp_m, tensor.arange(n_samples),:]

      if options.get('en_aux_inp',0):
          preact += tensor.dot(xAux,tparams[_p(prefix,'W_aux')])

      #  preact += tparams[_p(prefix, 'b')]
      h = [[]]*h_depth 
      c = [[]]*h_depth 
      
      for di in xrange(h_depth):
          i = tensor.nnet.sigmoid(sliceT(preact, 0, h_sz))
          f = tensor.nnet.sigmoid(sliceT(preact, 1, h_sz))
          o = tensor.nnet.sigmoid(sliceT(preact, 2, h_sz))
          c[di] = tensor.tanh(sliceT(preact, 3, h_sz))
          c[di] = f * sliceT(c_, di, h_sz) + i * c[di]
          h[di] = o * tensor.tanh(c[di])
          if di < (h_depth - 1):
              preact = tensor.dot(sliceT(h_, di+1, h_sz), tparams[_p(prefix, ('W_hid_' + str(di+1)))]) + \
                      tensor.dot(h[di], tparams[_p(prefix, ('W_inp_' + str(di+1)))])
      
      c_out = tensor.concatenate(c,axis=1)
      h_out = tensor.concatenate(h,axis=1)
      y = tensor.dot(h[-1],tparams['Wd']) + tparams['bd']
      xWIdx =  tensor.argmax(y, axis=-1,keepdims=True)
      xw_out = tparams['Wemb'][xWIdx.flatten()].reshape([n_samples,options['word_encoding_size']])

      return h_out, c_out, xw_out

  state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W_inp')]) + tparams[_p(prefix, 'b')])
  
  if options.get('en_aux_inp',0) == 0:
     aux_input = [] 
  
  if options.get('sched_sampling_mode',None) == None:
    sched_prob_mask = tensor.alloc(1, nsteps, n_samples)

  rval, updates = theano.scan(_step,
                              sequences=[state_below, sched_prob_mask],
                              outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                         n_samples,
                                                         h_depth*h_sz),
                                            tensor.alloc(numpy_floatX(0.),
                                                         n_samples,
                                                         h_depth*h_sz),
                                            tensor.alloc(numpy_floatX(0.),
                                                         n_samples,
                                                         options['word_encoding_size'])
                                            ],
                              non_sequences = [aux_input] ,
                              name=_p(prefix, '_layers'),
                              n_steps=nsteps)
  return rval, updates

# ======================== Dropout Layer =================================================
# Implements a simple dropout layer. When droput is on it drops units according to speeci-
# -fied prob and scales the rest. NOP otherwise 
# ========================================================================================
def dropout_layer(inp, use_noise, trng, prob, shp):
  scale = 1.0/(1.0-prob)
  proj = tensor.switch(use_noise,
                       (inp *
                        trng.binomial(shp,
                                      p=prob, n=1,
                                      dtype=inp.dtype)*scale),
                       inp)
  return proj


