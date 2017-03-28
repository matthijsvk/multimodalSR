from __future__ import print_function

import time
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

from lipreadingTCDTIMIT import *

def load_model (model_npz_file):
    if not os.path.exists(model_npz_file): print(
    "This npz file does not exist! Please run 'lipreadingTCDTIMIT' first to generate it.")

    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # activation
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    cnn = buildNetworks.build_network_resnet50(input)

    with np.load('./results/ResNet50/allLipspeakers/allLipspeakers.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(cnn['prob'], param_values)
    return cnn

def getPhonemeToVisemeMap():
    map = {'f':'A','v':'A',
            'er':'B','ow':'B','r':'B','q':'B','w':'B','uh':'B','uw':'B','axr':'B','ux':'B',
             'b':'C','p':'C','m':'C','em':'C',
             'aw':'D',
             ' dh':'E','th':'E',
             'ch':'F','jh':'F','sh':'F','zh':'F',
             'oy':'G', 'ao':'G',
             's':'H', 'z':'H',
             'aa':'I','ae':'I','ah':'I','ay':'I','ey':'I','ih':'I','iy':'I','y':'I','eh':'I','ax-h':'I','ax':'I','ix':'I',
             'd':'J','l':'J','n':'J','t':'J','el':'J','nx':'J','en':'J','dx':'J',
             'g':'K','k':'K','ng':'K','eng':'K',
             'sil':'S','pcl':'S','tcl':'S','kcl':'S','bcl':'S','dcl':'S','gcl':'S','h#':'S','#h':'S','pau':'S','epi':'S'
    }
    return map

def getPhonemeNumberMap (phonemeMap="./phonemeLabelConversion.txt"):
    phonemeNumberMap = {}
    with open(phonemeMap) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                phonemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = phoneme
                phonemeNumberMap[str(parts[1])] = parts[0]
    return phonemeNumberMap

def evaluateNetwork (X, y, model_npz_file):
    
    phonemeToViseme = getPhonemeToVisemeMap()
    phonemeNumberMap = getPhonemeNumberMap()  #bidirectional map phoneme-number
    for i in range(len(y)):
        y[i] = phonemeToViseme{phonemeNumberMap{y[i]}}  #viseme of the phoneme belonging to the y-number
        
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    
    cnn = load_network(model_npz_file)
    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)
    
    val_fn = theano.function([input, target], [test_loss, test_err])
    
    # calculate validation error of whole dataset
    err = 0
    loss = 0
    batches = len(X) / batch_size
    
    for i in range(batches):
        new_loss, new_err = val_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
        err += new_err
        loss += new_loss
    
    val_err = err / batches * 100
    val_loss /= batches

    print("  validation loss:               " + str(val_loss))
    print("  validation error rate:         " + str(val_err) + "%")
    print("  test loss:                     " + str(test_loss))
    print("  test error rate:               " + str(test_err) + "%")