import numpy as np
import theano
import theano.tensor as T

batch_size = 32
Tmask = T.imatrix()
eqs = T.neq(Tmask,T.zeros(Tmask.shape))
f = theano.function([Tmask],eqs)
indices = eqs.nonzero()
g = theano.function([Tmask], indices)

mask = np.array([np.array([0, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])],dtype='int32')
print(mask)
fm = f(mask); print(fm)
gm = g(mask); print(gm)

import pdb;pdb.set_trace()

# y=T.fmatrix()
#
# y01x = y.dimshuffle(0, 1, 'x')
# y0x1 = y.dimshuffle(0, 'x', 1)
# yx01 = y.dimshuffle('x', 0, 1)
#
# f01x = theano.function(inputs=[y], outputs=y01x)
# f0x1 = theano.function(inputs=[y], outputs=y0x1)
# fx01 = theano.function(inputs=[y], outputs=yx01)
#
# print f01x(np.array(np.random.rand(10, 2), dtype=np.float32)).shape
# print f0x1(np.array(np.random.rand(10, 2), dtype=np.float32)).shape
# print fx01(np.array(np.random.rand(10, 2), dtype=np.float32)).shape