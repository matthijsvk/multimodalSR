'''
Feed-forward temporal integration layers.
'''
import lasagne
import theano.tensor as T


class AttentionLayer(lasagne.layers.Layer):
    '''
    A layer which computes a weighted average across the second dimension of
    its input, where the weights are computed according to the third dimension.
    This results in the second dimension being flattened.  This is an attention
    mechanism - which "steps" (in the second dimension) are attended to is
    determined by a learned transform of the features.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs,).

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be () (it is a scalar).
        If None is provided the layer will have no biases.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    '''
    def __init__(self, incoming, W=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        # Use identity nonlinearity if provided nonlinearity is None
        self.nonlinearity = (lasagne.nonlinearities.identity
                             if nonlinearity is None else nonlinearity)

        # Add weight vector parameter
        self.W = self.add_param(W, (self.input_shape[2],), name="W")
        if b is None:
            self.b = None
        else:
            # Add bias scalar parameter
            self.b = self.add_param(b, (), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_output_for(self, input, **kwargs):
        # Dot with W to get raw weights, shape=(n_batch, n_steps)
        activation = T.dot(input, self.W)
        # Add bias
        if self.b is not None:
            activation = activation + self.b
        # Apply nonlinearity
        activation = self.nonlinearity(activation)
        # Perform softmax
        activation = T.exp(activation)
        activation /= activation.sum(axis=1).dimshuffle(0, 'x')
        # Weight steps
        weighted_input = input*activation.dimshuffle(0, 1, 'x')
        # Compute weighted average (summing because softmax is normed)
        return weighted_input.sum(axis=1)


class MeanLayer(lasagne.layers.Layer):
    '''
    A layer which computes an average across the second dimension of
    its input. This results in the second dimension being flattened.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    '''
    def __init__(self, incoming, **kwargs):
        super(MeanLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_output_for(self, input, **kwargs):
        # Compute average of second axis
        return input.mean(axis=1)
