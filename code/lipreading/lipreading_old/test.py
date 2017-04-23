from __future__ import print_function

import sys
import os
import time

import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data = unpickle(os.path.join(os.path.expanduser('~/TCDTIMIT/database_binaryViseme/Lipspkr1.pkl')))
print(data.keys())
print(data)

thisN = data['data'].shape[0]
print("This dataset contains ", thisN, " images")

