import numpy as np
import theano
import theano.tensor as T

# Fn=Fn−2+Fn−1, with F1=1 and F2=1.

f1 = theano.shared(1)
f2 = theano.shared(1)

f = f1 + f2
updates= {a:b; b:f}
next = theano.function([f1,f2],f,update=updates)
[next() for i in range(0,3)]
