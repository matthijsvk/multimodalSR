import numpy as np
import theano
import argparse
import json
import os
from utils.solver import Solver 
from utils.dataprovider import DataProvider
from utils.utils import getModelObj

def main(params):
  
  # main training and validation loop goes here
  # This code should be independent of which model we use
  batch_size = params['batch_size']
  max_epochs = params['max_epochs']
  
  
  # fetch the data provider object
  dp = DataProvider(params)
  params['feat_size'] = dp.feat_size
  params['phone_vocab_size'] = dp.phone_vocab
  # Get the solver object, optional not needed for kerras
  # solver = Solver(params['solver'])
  ## Add the model intiailization code here
  
  modelObj = getModelObj(params)

  # Build the model Architecture
  f_train = modelObj.build_model(params)
  
  if params['saved_model'] !=None: 
    cv = json.load(open(params['saved_model'],'r'))
    modelObj.model.load_weights(cv['weights_file'])
    print 'Conitnuing training from model %s'%(params['saved_model'])
  
  train_x, train_y, val_x, val_y = dp.get_data_array(params['model_type'], ['train', 'devel'], cntxt=params['context'])
  fname, best_val_loss = modelObj.train_model(train_x, train_y, val_x, val_y, params)

  checkpoint = {}
    
  checkpoint['params'] = params
  checkpoint['weights_file'] = fname.format(val_loss=best_val_loss)
  filename = 'model_%s_%s_%s_%.2f.json' % (params['dataset'], params['model_type'], params['out_file_append'], best_val_loss)
  filename = os.path.join(params['out_dir'],filename)
  print 'Saving to File %s'%(filename)
  json.dump(checkpoint, open(filename,'w'))

  ## Now let's build a gradient computation graph and rmsprop update mechanism
  ##grads = tensor.grad(cost, wrt=model.values())
  ##lr = tensor.scalar(name='lr',dtype=config.floatX)
  ##f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, grads,
  ##                                   inp_list, cost, params)

  #num_frames_total = dp.getSplitSize('train')
  #num_iters_one_epoch = num_frames_total/ batch_size
  #max_iters = max_epochs * num_iters_one_epoch
  ##
  #for it in xrange(max_iters):
  #  batch = dp.getBatch(batch_size)
  #  cost = f_train(*batch)
    
    #cost = f_grad_shared(inp_list)
    #f_update(params['learning_rate'])

    #Save model periodically
  return modelObj


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  # IO specs
  parser.add_argument('-d','--dataset', dest='dataset', type=str, default='mvNorm', help='Which file should we use for read the MFCC features')
  parser.add_argument('--dataset_desc', dest='dataDesc', type=str, default='dataset.json', help='Which file should we use for read the MFCC features')
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='data/default_feats.p', help='Which file should we use for read the MFCC features')
  parser.add_argument('--output_file_append', dest='out_file_append', type=str, default='dummyModel', help='String to append to the filename of the trained model')

  parser.add_argument('--out_dir', dest='out_dir', type=str, default='cv/', help='String to append to the filename of the trained model')
  parser.add_argument('--in_dim', dest='in_dim', type=int, default=39, help='Input dimension')
  
  # Learning related parmeters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=20, help='number of epochs to train for')
  parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=1e-1, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  parser.add_argument('-pl', '--pre_learning_rate', dest='plr', type=float, default=1e-1, help='solver pre-learning rate')  
  parser.add_argument('-pm', '--pre_max_epochs', dest='pre_max_epochs', type=int, default=20, help='number of epochs to pre-train for')
  
  # Solver related parameters
  parser.add_argument('--solver', dest='solver', type=str, default='sgd', help='solver types supported: rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.9999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  
  
  # Model architecture related parameters
  parser.add_argument('--model_type', dest='model_type', type=str, default='MLP', help='Can take values MLP, RNN or LSTM')
  parser.add_argument('--recurrent_type', dest='recurrent_type', type=str, default='simple', help='Can take values MLP, RNN or LSTM')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1, help='enable or disable dropout')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.0, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--hidden_layers', dest='hidden_layers', nargs='+',type=int, default=[300, 300], help='the hidden layer configuration, for applicable to MLP')
  
  # RNN Model architecture related parameters
  parser.add_argument('--context', dest='context', type=int, default=-1, help='context before current sample to feed to RNN')
  
  parser.add_argument('--continue_training', dest='saved_model', type=str, default=None, help='input the saved model json file to evluate on')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  model = main(params)
