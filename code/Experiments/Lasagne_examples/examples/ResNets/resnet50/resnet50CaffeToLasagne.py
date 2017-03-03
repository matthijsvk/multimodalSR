# this gets a caffe network as input, and returns a Lasagne network as output


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

import caffe
import numpy as np
import pickle

from resnet50LasagneModel import * # here the Resnet50 Lasagne model is described

#### Gathering everything together
def build_network_fill_from_caffe():  # uses the model structure from build_model, fills the parameters from './ResNet-50-deploy.prototxt', './ResNet-50-model.caffemodel'
    
    # First, create the Lasagne Resnet50. Afterward, transfer weights from the caffe model.
    # Create head of the network (everything before first residual block) in Lasagne
    net = build_model()
    print 'Number of Lasagne layers:', len(lasagne.layers.get_all_layers(net['prob']))
    
    # # Transfer weights from caffe to lasagne
    # ## Load pretrained caffe model
    net_caffe = caffe.Net('./modelFiles/ResNet_TrainedModel/ResNet-50-deploy.prototxt', './modelFiles/ResNet_TrainedModel/ResNet-50-model.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
    print 'Number of caffe layers: %i' % len(layers_caffe.keys())
    
    # ## Copy weights
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
            layer.inv_std.set_value(1 / np.sqrt(layer_bn_caffe.blobs[1].data) + 1e-4)
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


# These functions are for using the Resnet50 for ImageNet, but can be used for
# needed for evaluating the model
def get_classes (filename='./modelFiles/imagenet_classes.txt'):
    print("getting classes...")
    with open(filename, 'r') as f:
        classes = map(lambda s: s.strip(), f.readlines())
    return classes

def get_mean_values (filename='./modelFiles/ResNet_TrainedModel/ResNet_mean.binaryproto'):
    # Load mean values
    print("loading mean values...")
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(filename, 'rb').read()
    blob.ParseFromString(data)
    mean_values = np.array(caffe.io.blobproto_to_array(blob))[0]
    
    return mean_values


if __name__ == "__main__":
    net, net_caffe = build_network_fill_from_caffe()  # from resnetCaffeToLasagne.py, builds network and then fills it with the Caffe weights
    
    print("getting classes...")
    classes = get_classes()  # put filename here, default = "./imagenet_classes.txt"
    
    print("getting mean_values...")
    mean_values = get_mean_values()  # put filename here, default = './ResNet_mean.binaryproto'
    
    # print("testing model...")
    # print("getting image urls...")      #put url here, default = 'http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html'. Should be a list of image urls to download, open this link for an example
    # image_urls = download_images()
    # print("evaluating...")
    # test_lasagne(classes, image_urls, mean_values, net, net_caffe)
    
    model = {
        'values': lasagne.layers.get_all_param_values(net['prob']),
        'synset_words': classes,
        'mean_image': mean_values
    }
    
    print "Compared all images. Storing Lasagne model in Pickle (pkl) file..."
    pickle.dump(model, open('./resnet50imageNet.pkl', 'wb'), protocol=-1)