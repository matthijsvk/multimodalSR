import numpy as np

a =[ 'a','b','c','d']
b = [1,2,3,4]
c = ['one', 'two', 'three', 'four']
d = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
lst = [a,b,c,d]

def shuffle(lst):
    import random
    c = list(zip(*lst))
    random.shuffle(c)
    shuffled = zip(*c)
    for i in range(len(shuffled)):
        shuffled[i] = list(shuffled[i])
    return shuffled


a,b,c,d = shuffle(lst)

import pdb;pdb.set_trace()

import random

a = ['a', 'b', 'c']
b = [1, 2, 3]

c = list(zip(a, b))

random.shuffle(c)

a, b = zip(*c)

print a
print b