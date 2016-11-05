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

parser = argparse.ArgumentParser(description="Getting top 5 classes of images")

add_arg = parser.add_argument

add_arg("-i", "--input_image", help="Input image")
add_arg("-m", "--model_file", help="Model pickle file")

args = parser.parse_args()

import resnet50

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
	net, mean_img, synset_words = resnet50.load_model(pkl_model)

	get_class_prob = theano.function([net['input'].input_var], lasagne.layers.get_output(net['prob'],deterministic=True))

	def print_top5(im_path):
		raw_im, im = prep_image(im_path, mean_img)
		prob = get_class_prob(im)[0]
		res = sorted(zip(synset_words, prob), key=lambda t: t[1], reverse=True)[:5]
		for c, p in res:
			print '  ', c, p

	return get_class_prob, print_top5

def get_feature_extractor(pkl_model, layer_name):
	net, mean_img, synset_words = resnet50.load_model(pkl_model)
	layer_output = theano.function([net['input'].input_var], lasagne.layers.get_output(net[layer_name],deterministic=True))

	def feature_extractor(im_path):
		raw_im, im = prep_image(im_path, mean_img)
		return layer_output(im)[0]

	return feature_extractor

if __name__ == "__main__":
	print "Compiling functions..."
	get_prob, print_top5 = get_net_fun(args.model_file)
	t0 = time.time()
	print_top5(args.input_image)
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))

	print "Compiling function for getting conv1 ...."
	feature_extractor = get_feature_extractor(args.model_file, 'conv1')
	t0 = time.time()
	print feature_extractor(args.input_image).shape
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))

	print "Compiling function for getting res2c ...."
	feature_extractor = get_feature_extractor(args.model_file, 'res2c')
	t0 = time.time()
	print feature_extractor(args.input_image).shape
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))

	print "Compiling function for getting res3d ...."
	feature_extractor = get_feature_extractor(args.model_file, 'res3d')
	t0 = time.time()
	print feature_extractor(args.input_image).shape
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))

	print "Compiling function for getting conv res4f ...."
	feature_extractor = get_feature_extractor(args.model_file, 'res4f')
	t0 = time.time()
	print feature_extractor(args.input_image).shape
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))

	print "Compiling function for getting conv res5c ...."
	feature_extractor = get_feature_extractor(args.model_file, 'res5c')
	t0 = time.time()
	print feature_extractor(args.input_image).shape
	t1 = time.time()
	print("Total time taken {:.4f}".format(t1 - t0))




