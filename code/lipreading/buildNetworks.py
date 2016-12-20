# ResNet-50, network from the paper:
# "Deep Residual Learning for Image Recognition"
# http://arxiv.org/pdf/1512.03385v1.pdf
# License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_network_resnet50(input):
    net = {}
    net['input'] = InputLayer(shape=(None, 1, 120, 120),input_var=input)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=39, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    return net

# network from google BBC paper
def build_network_google (activation, alpha, epsilon, input):
    # input
    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 120, 120),  # 5,120,120 (5 = #frames)
            input_var=input)
    # conv 1
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # conv 2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            stride=(2, 2),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # conv3
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # conv 4
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # conv 5
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # FC layer
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=39)

    # cnn = lasagne.layers.BatchNormLayer(
    #       cnn,
    #       epsilon=epsilon,
    #       alpha=alpha)
    return cnn


# default network for cifar10
def build_network_cifar10 (activation, alpha, epsilon, input):
    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 120, 120),
            input_var=input)
    
    # 128C3-128C3-P2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # 256C3-256C3-P2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # 512C3-512C3-P2
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    cnn = lasagne.layers.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)
    
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    # print(cnn.output_shape)
    
    # 1024FP-1024FP-10FP
    #cnn = lasagne.layers.DenseLayer(
    #        cnn,
    #        nonlinearity=lasagne.nonlinearities.identity,
    #        num_units=1024)
    
   # cnn = lasagne.layers.BatchNormLayer(
   #         cnn,
   #         epsilon=epsilon,
   #         alpha=alpha)
    
   # cnn = lasagne.layers.NonlinearityLayer(
   #         cnn,
   #         nonlinearity=activation)
   # 
   # cnn = lasagne.layers.DenseLayer(
   #         cnn,
   #         nonlinearity=lasagne.nonlinearities.identity,
   #         num_units=1024)
    
   # cnn = lasagne.layers.BatchNormLayer(
   #         cnn,
   #        # epsilon=epsilon,
   #         alpha=alpha)
    
   # cnn = lasagne.layers.NonlinearityLayer(
   #         cnn,
   #         nonlinearity=activation)
    
    cnn = lasagne.layers.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=39)
    
    #cnn = lasagne.layers.BatchNormLayer(
    #        cnn,
    #        epsilon=epsilon,
    #        alpha=alpha)
    
    return cnn
