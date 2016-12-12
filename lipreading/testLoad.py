import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

batch1 = unpickle("/home/matthijs/TCDTIMIT/database_binary/Lipspkr1.pkl")

print(batch1['data'][0:10])

ys = batch1['labels'][0:9]
if isinstance(ys, list):
    ys = np.asarray(ys).astype('uint8')
    
print(ys)
ys = ys -1
print(ys)
