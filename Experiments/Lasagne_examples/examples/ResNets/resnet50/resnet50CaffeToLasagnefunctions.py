
#### Gathering everighting together
def build_model: # needs the files './ResNet-50-deploy.prototxt', './ResNet-50-model.caffemodel'
    # Create head of the network (everithing before first residual block)
    
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']
    
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    
    # Create four groups of residual blocks
    
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)
    
    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4,
                                                              ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)
    
    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4,
                                                              ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)
    
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4,
                                                              ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
        
    # Create tail of the network (everighting after last residual block)
    
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
    
    print 'Number of Lasagne layers:', len(lasagne.layers.get_all_layers(net['prob']))

    # # Transfer weights from caffe to lasagne
    # ## Load pretrained caffe model

    net_caffe = caffe.Net('./ResNet-50-deploy.prototxt', './ResNet-50-model.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
    print 'Number of caffe layers: %i' % len(layers_caffe.keys())

    # ## Copy weights
    #
    # There is one more issue with BN layer: caffa stores variance $\sigma^2$, but lasagne stores inverted standard deviation $\dfrac{1}{\sigma}$, so we need make simple transfommation to handle it.
    # Other issue reffers to weights ofthe dense layer, in caffe it is transposed, we should handle it too.

    for name, layer in net.items():
        if name not in layers_caffe:
            print name, type(layer).__name__
            continue
        if isinstance(layer, BatchNormLayer):
            layer_bn_caffe = layers_caffe[name]
            layer_scale_caffe = layers_caffe['scale' + name[2:]]
            layer.gamma.set_value(layer_scale_caffe.blobs[0].data)
            layer.beta.set_value(layer_scale_caffe.blobs[1].data)
            layer.mean.set_value(layer_bn_caffe.blobs[0].data)
            layer.inv_std.set_value(1/np.sqrt(layer_bn_caffe.blobs[1].data) + 1e-4)
            continue
        if isinstance(layer, DenseLayer):
            layer.W.set_value(layers_caffe[name].blobs[0].data.T)
            layer.b.set_value(layers_caffe[name].blobs[1].data)
            continue
        if len(layers_caffe[name].blobs) > 0:
            layer.W.set_value(layers_caffe[name].blobs[0].data)
        if len(layers_caffe[name].blobs) > 1:
            layer.b.set_value(layers_caffe[name].blobs[1].data)
    
    # now, a Lasagne network is created and stored in the 'net' variable
    # the Caffe model ist stored in 'net_caffe'

    return net, net_caffe