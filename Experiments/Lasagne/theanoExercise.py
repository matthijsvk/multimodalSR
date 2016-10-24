import theano
from theano import tensor as T

# # functions
# a = T.vector() # declare variable
# b = T.vector()
# out = a**2 + b**2 + 2*a*b               # build symbolic expression
# f = theano.function([a,b], out)   # compile function
# print(f([1],[2]))
#
# # computing several things at same time
# a, b = T.dmatrices('a', 'b')
# diff = a - b
# abs_diff = abs(diff)
# diff_squared = diff**2
# f = theano.function([a, b], [diff, abs_diff, diff_squared])

# # default argument
# f = function([x, In(y, value=1)], z)