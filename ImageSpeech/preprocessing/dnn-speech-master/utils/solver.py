import time
import numpy as np
import theano
import theano.tensor as tensor
from utils import numpy_floatX

class Solver:
  """
  solver worries about:
  - different optimization methods, updates, weight decays
  """
  def __init__(self,solver):
    if solver == 'rmsprop':
        self.build_solver_model = self.rmsprop
    else:
        raise ValueError('ERROR: %s --> This solver type is not yet supported'%(solver))

# ========================================================================================
  def rmsprop(self, lr, tparams, grads, inp_list, cost, params):
    clip = params['grad_clip']
    decay_rate = tensor.constant(params['decay_rate'], dtype=theano.config.floatX) 
    smooth_eps = tensor.constant(params['smooth_eps'], dtype=theano.config.floatX)
    zipped_grads = [theano.shared(np.zeros_like(p.get_value()),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(np.zeros_like(p.get_value()),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    if clip > 0.0:
        rg2up = [(rg2, tensor.clip(decay_rate * rg2 + (1 - decay_rate) * (tensor.clip(g,-clip,clip) ** 2),0.0,np.inf))
             for rg2, g in zip(running_grads2, grads)]
    else:
        rg2up = [(rg2, tensor.clip(decay_rate * rg2 + (1 - decay_rate) * (g ** 2),0.0,np.inf))
             for rg2, g in zip(running_grads2, grads)]
  
    f_grad_shared = theano.function(inp_list, cost,
                                    updates=zgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, -lr * zg / (tensor.sqrt(rg2)+ smooth_eps))
                 for ud, zg, rg2 in zip(updir, zipped_grads, 
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update, zipped_grads, running_grads2, updir
