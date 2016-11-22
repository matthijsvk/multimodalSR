from resnet50CaffeToLasagne import * #function to build Lasagne model from the Caffe model and transfer weights: build_network_fill_from_caffe()

import lasagne
import caffe

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer # can be replaced with dnn layers
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax


# Helper modules, some of them will help us to download images and plot them
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 6
import io
import urllib
import skimage.transform
import pickle


# needed for evaluating the model
def get_classes(filename = './imagenet_classes.txt'):
    print("getting classes...")
    with open(filename, 'r') as f:
        classes = map(lambda s: s.strip(), f.readlines())
        
    return classes

def get_mean_values(filename='./ResNet_mean.binaryproto'):
    # Load mean values
    print("loading mean values...")
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(filename, 'rb').read()
    blob.ParseFromString(data)
    mean_values = np.array(caffe.io.blobproto_to_array(blob))[0]
    
    return mean_values


# just for testing
def download_images(url='http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html'):
    # Read ImageNet synset

    # Download some image urls for recognition
    print("getting urls...")
    index = urllib.urlopen(url).read()
    image_urls = index.split('<br>')
    np.random.seed(23)
    np.random.shuffle(image_urls)
    image_urls = image_urls[:10]  #used to be 100
    
    return image_urls

# Image loader
def prep_image (url, mean_values, fname=None):
    if fname is None:
        ext = url.split('.')[-1]
        im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    else:
        ext = fname.split('.')[-1]
        im = plt.imread(fname, ext)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]
    rawim = np.copy(im).astype('uint8')
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im[::-1, :, :]
    im = im - mean_values
    return rawim, floatX(im[np.newaxis])

# Lets take five images and compare prediction of Lasagne with Caffe
def test_lasagne(classes, image_urls, mean_values, net, net_caffe):
    n = 5
    m = 5
    i = 0
    for url in image_urls:
        print url  ### print url to show progress ###
        # try:
        rawim, im = prep_image(url, mean_values)
        # except:
        #     print 'Failed to download'
        #     continue
        
        prob_lasangne = np.array(lasagne.layers.get_output(net['prob'], im, deterministic=True).eval())[0]
        prob_caffe = net_caffe.forward_all(data=im)['prob'][0]
        
        print 'Lasagne:'
        res = sorted(zip(classes, prob_lasangne), key=lambda t: t[1], reverse=True)[:n]
        for c, p in res:
            print '  ', c, p
        
        print 'Caffe:'
        res = sorted(zip(classes, prob_caffe), key=lambda t: t[1], reverse=True)[:n]
        for c, p in res:
           print '  ', c, p
        
        plt.figure()
        plt.imshow(rawim.astype('uint8'))
        plt.axis('off')
        plt.show()
        
        i += 1
        if i == m:
            break
        
        print '\n\n'


if __name__ == "__main__":
    
    net, net_caffe = build_network_fill_from_caffe() # from resnetCaffeToLasagne.py, builds network and then fills it with the Caffe weights
    
    print("getting classes...")
    classes = get_classes()             #put filename here, default = "./imagenet_classes.txt"
    
    print("getting mean_values...")
    mean_values = get_mean_values()     #put filename here, default = './ResNet_mean.binaryproto'

    # print("testing model...")
    # print("getting image urls...")      #put url here, default = 'http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html'. Should be a list of image urls to download, open this link for an example
    #image_urls = download_images()
    # print("evaluating...")
    #test_lasagne(classes, image_urls, mean_values, net, net_caffe)
    
    model = {
        'values': lasagne.layers.get_all_param_values(net['prob']),
        'synset_words': classes,
        'mean_image': mean_values
    }

    print "comapared all images. Storing Lasagne model in Pickle (pkl) file..."
    pickle.dump(model, open('./resnet50imageNet.pkl', 'wb'), protocol=-1)