import cPickle as pickle
import json
import numpy as np
import os

class DataProvider:
  def __init__(self, params):
    # Write the initilization code to load the preprocessed data and labels
    self.dataDesc = json.load(open(os.path.join('data', params['dataset'], params['dataDesc']), 'r'))
    self.in_dim = params['in_dim']
    self.data = {}
    for splt in ['train','eval','devel']:
      self.data[splt] = {}
      self.data[splt]['feat'],self.data[splt]['lab'] = self.load_data(self.dataDesc[splt+'_x'], self.dataDesc[splt+'_y'])

    self.feat_size = self.in_dim 
    self.phone_vocab = len(self.dataDesc['ph2bin'].keys())

  def getBatch(self, batch_size):
    return []*batch_size

  def getBatchWithContext(self):
    return []
  
  def get_data_array(self, model, splits, cntxt=-1, shufdata=1, idx = -1):
    output = []

    for spt in splits:
        if model == 'MLP' or model == 'DBN':
            final_feats = self.data[spt]['feat'] if idx == -1 else [self.data[spt]['feat'][idx]]
        elif model == 'RNN':
            inp_feats = self.data[spt]['feat'] if idx == -1 else [self.data[spt]['feat'][idx]]
            final_feats = [] 
            for feat in inp_feats:
                padFeat = np.concatenate([np.zeros((cntxt-1,self.feat_size)), feat])
                idces = np.repeat(np.arange(cntxt-1,padFeat.shape[0]),cntxt) + np.tile(np.arange(-(cntxt-1),1),padFeat.shape[0]- cntxt +1)
                cntxtDat = padFeat[idces,:].reshape(feat.shape[0], cntxt, self.feat_size)
                final_feats.append(cntxtDat)
            
        feats = np.concatenate(final_feats)
        labs = self.data[spt]['lab'] if idx == -1 else [self.data[spt]['lab'][idx]]
        labs = np.concatenate(labs)
        shfidx = np.random.permutation(feats.shape[0]) if shufdata == 1 else np.arange(feats.shape[0])
        feats = feats[shfidx,...]
        labs = labs[shfidx,:]
        output.extend([feats,labs])
        
    return output

  def getSplitSize(self, split='train'):
    return self.data[split]['feat'].shape[0] 
  
  def getSplitSize(self, split='train'):
    return self.data[split]['feat'].shape[0] 

  def load_data(self, input_file_list, output_file_list, out_dim=24, shufdata = 0):
      """
      load partition
      """
      in_dim = self.in_dim
      input_data = []
      output_data = []
      for i in xrange(len(input_file_list)):  
          in_data = np.fromfile(input_file_list[i],dtype=np.float32,sep=' ',count=-1)
          out_data = np.fromfile(output_file_list[i],dtype=np.float32,sep=' ',count=-1)
          in_data.resize(len(in_data)/in_dim, in_dim)
          out_data.resize(len(out_data)/out_dim, out_dim)
          input_data.append(in_data)
          output_data.append(out_data)
  
      return input_data, output_data

