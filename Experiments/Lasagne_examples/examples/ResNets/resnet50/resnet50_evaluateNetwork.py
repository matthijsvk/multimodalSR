import urllib
import io
import skimage.transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 6
import argparse
import time
import pickle
import numpy as np
import theano
import lasagne
import os


parser = argparse.ArgumentParser(description="Getting top 5 classes of images")

add_arg = parser.add_argument

add_arg("-i", "--input_image", help="Input image")
add_arg("-m", "--model_file", help="Model pickle file")

args = parser.parse_args()

# this imported file contains build_model(), which constructs the network structure that you van fill using the pkl file
# to generate the pkl file, you need to run the main function in resnet50CaffeToLasagne_ImageNet,
#   which populates the network from caffe, gets the classes and the mean image, and stores those in a pkl file
from resnet50LasagneModel import *

def load_model(model_pkl_file):
    if not os.path.exists(model_pkl_file): print("This pkl file does not exist! Please run 'resnet50CaffeToLasagne' first to generate it.")
    model = pickle.load(open(model_pkl_file,'rb'))
    net = build_model()
    lasagne.layers.set_all_param_values(net['prob'], model['values'])
    return net, model['mean_image'], model['synset_words']

def prep_image(fname, mean_values):
    t0 = time.time()
    ext = fname.split('.')[-1]
    im = plt.imread(fname, ext)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    # h, w, _ = im.shape
    # im = skimage.transform.resize(im, (224, 224), preserve_range=True)
    h, w, _ = im.shape
    rawim = np.copy(im).astype('uint8')
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im[::-1, :, :]
    im = im - mean_values
    t1 = time.time()
    print "Time taken in preparing the image : {}".format(t1 - t0)
    return rawim, im[np.newaxis].astype('float32')

def get_net_fun(pkl_model):
	net, mean_img, synset_words = load_model(pkl_model)

	get_class_prob = theano.function([net['input'].input_var], lasagne.layers.get_output(net['prob'],deterministic=True))

	def print_top5(im_path):
		raw_im, im = prep_image(im_path, mean_img)
		prob = get_class_prob(im)[0]
		res = sorted(zip(synset_words, prob), key=lambda t: t[1], reverse=True)[:5]
		for c, p in res:
			print '  ', c, p

	return get_class_prob, print_top5

def get_feature_extractor(pkl_model, layer_name):
	net, mean_img, synset_words = load_model(pkl_model)
	layer_output = theano.function([net['input'].input_var], lasagne.layers.get_output(net[layer_name],deterministic=True))

	def feature_extractor(im_path):
		raw_im, im = prep_image(im_path, mean_img)
		return layer_output(im)[0]

	return feature_extractor


# The following functions are just for testing out (with ImageNet examples).
def download_images (url='http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html'):
    # Read ImageNet synset
    
    # Download some image urls for recognition
    print("getting urls...")
    index = urllib.urlopen(url).read()
    image_urls = index.split('<br>')
    np.random.seed(23)
    np.random.shuffle(image_urls)
    image_urls = image_urls[:10]  # used to be 100
    
    return image_urls

# Lets take five images and compare prediction of Lasagne with Caffe
def test_lasagne_ImageNet (classes, image_urls, mean_values, net, net_caffe):
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


if __name__ == "__main__":
	print "Compiling functions..."
	get_prob, print_top5 = get_net_fun(args.model_file) # expects pkl model
	t0 = time.clock()
	print_top5(args.input_image)
	t1 = time.clock()
	print("Total time taken {:.4f}".format(t1 - t0))

	# print "Compiling function for getting conv1 ...."
	# feature_extractor = get_feature_extractor(args.model_file, 'conv1')
	# t0 = time.time()
	# print feature_extractor(args.input_image).shape
	# t1 = time.time()
	# print("Total time taken {:.4f}".format(t1 - t0))
    #
	# print "Compiling function for getting res2c ...."
	# feature_extractor = get_feature_extractor(args.model_file, 'res2c')
	# t0 = time.time()
	# print feature_extractor(args.input_image).shape
	# t1 = time.time()
	# print("Total time taken {:.4f}".format(t1 - t0))
    #
	# print "Compiling function for getting res3d ...."
	# feature_extractor = get_feature_extractor(args.model_file, 'res3d')
	# t0 = time.time()
	# print feature_extractor(args.input_image).shape
	# t1 = time.time()
	# print("Total time taken {:.4f}".format(t1 - t0))
    #
	# print "Compiling function for getting conv res4f ...."
	# feature_extractor = get_feature_extractor(args.model_file, 'res4f')
	# t0 = time.time()
	# print feature_extractor(args.input_image).shape
	# t1 = time.time()
	# print("Total time taken {:.4f}".format(t1 - t0))
    # 
	# print "Compiling function for getting conv res5c ...."
	# feature_extractor = get_feature_extractor(args.model_file, 'res5c')
	# t0 = time.time()
	# print feature_extractor(args.input_image).shape
	# t1 = time.time()
	# print("Total time taken {:.4f}".format(
    
    
    
# Usage examples
# first, generate the pkl model: 'python resnet50CaffeToLasagne.py'
# then, evaluate the model:      'python resnet50_evaluateNetwork.py -i indianElephant.jpeg -m resnet50imageNet.pkl'
#  -> this gives the 5 most probable classes of the image 'indianElephant.jpeg'
