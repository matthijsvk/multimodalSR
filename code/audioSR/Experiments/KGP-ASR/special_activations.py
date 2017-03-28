import theano
import theano.tensor as T 


def clipped_relu(x, clipping_value = 20):
    """
    This activation function has been found to work well with deep Bidirectional RNNs 
    and helps in preventing overflow.
    It implements:
        
        y = min(max(0,x),clipping_value)

    x: input theano tensor
    clipping value: scalar float
    """
    return T.switch(x>clipping_value, clipping_value, T.nnet.relu(x))