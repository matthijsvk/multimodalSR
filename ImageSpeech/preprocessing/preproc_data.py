import os
import sys
import numpy as np
import theano
import json
import argparse
from windowing import sliding_window

def read_file_list(files_name):
  """
  convert the to file list
  """
  files_list = []
  fid = open(files_name)
  for line in fid.readlines():
      line = line.strip()
      if len(line) < 1:
          continue
      files_list.append(line)
  fid.close()
  return files_list


def extract_file_id_list(files_list):
  """
  remove any file extensions
  """
  files_id_list = []
  for file_name in files_list:
      file_id = os.path.basename(os.path.splitext(file_name)[0])
      files_id_list.append(file_id)

  return  files_id_list


def prepare_file_path_list(files_id_list, files_dir, files_extension, 
  new_dir_switch=True):
  if not os.path.exists(files_dir) and new_dir_switch:
      os.makedirs(files_dir)
  files_name_list = []
  for file_id in files_id_list:
      file_name = os.path.join(files_dir, file_id + files_extension)
      files_name_list.append(file_name)

  return  files_name_list


def phoneme_binary(phoneme_list):
  n = len(phoneme_list)
  binary = ['0']*n
  phoneme_bin = {}
  for i in xrange(n):
      ph = phoneme_list[i]
      binary[i] = '1'
      phoneme_bin[ph] = ' '.join(binary)
      binary[i] = '0'

  return phoneme_bin


def lab2binary(in_file_list, ph_bin, out_file_list):
  """
  in_file_list: list of label files
  ph_bin: dictionary for phoneme to binary label
  """
  for i in xrange(len(in_file_list)):
      with open(in_file_list[i],'r') as lab:
          b_label = [ph_bin[l.strip()] for l in lab.readlines()]
      
      with open(out_file_list[i], 'w') as outlab:
          outlab.write('\n'.join(b_label))


def make_context_frames(in_file_list, out_file_list, n_frames=3, mfcc_dim=39):
  """
  this function makes the input frames into context dependent
  """
  win_length = n_frames*mfcc_dim
  shift_length = mfcc_dim
  temp = np.zeros(((n_frames-1)*mfcc_dim))

  for i in xrange(len(in_file_list)):
      data = np.fromfile(in_file_list[i], dtype=np.float32, sep=' ', count=-1)
      # append zeros in the beginning
      data = np.hstack((temp, data))
      new_data = sliding_window(data, win_length, shift_length,flatten=False)
      
      np.savetxt(out_file_list[i], new_data, fmt='%.6f', delimiter=' ')


def MVN_normalize(in_file_list, mfcc_dim, out_file_list):
  """
  mean and variance normalization
  """
  for i in xrange(len(in_file_list)):
      data = np.fromfile(in_file_list[i], dtype=np.float32, sep=' ', count=-1)
      data.resize(len(data)/mfcc_dim, mfcc_dim)
      data_mean = np.mean(data, axis=0)
      data_std = np.std(data, axis=0)
      data = (data - np.tile(data_mean, (data.shape[0],1)))/np.tile(data_std, (data.shape[0],1))

      np.savetxt(out_file_list[i], data, fmt='%.6f', delimiter=' ')


def main(params):
  data_dir = params['data_dir']

  srcDir = os.path.join(data_dir, params['src_dir'])
  destDir = os.path.join(data_dir, params['dest_dir'])

  splits = ['train','eval','devel']
  data_desc = {}

  phone_set = set()
  for splt in splits:
    sp_srcDir = os.path.join(srcDir, splt)
    sp_destDir = os.path.join(destDir, splt)
    files_id_list = []
    for fname in os.listdir(sp_srcDir):
        if fname.endswith('.mfcc'):
            files_id_list.append(os.path.basename(os.path.splitext(fname)[0]))
        # phone vocabulary building should be only done on train set
        if splt ==  'train':
            if fname.endswith('.labels'):
                phone_set.update(set(open(os.path.join(sp_srcDir,fname),'r').read().splitlines()))
            
            
    labels_file_list = prepare_file_path_list(files_id_list,
                                              sp_srcDir, '.labels', False)
    binary_labels_file_list = prepare_file_path_list(files_id_list,
                                          sp_destDir, '.labels')
    mfcc_file_list = prepare_file_path_list(files_id_list,
                                          sp_srcDir, '.mfcc', False)
    norm_mfcc_file_list = prepare_file_path_list(files_id_list,
                                          sp_destDir, '.mfcc')
    window_mfcc_file_list = prepare_file_path_list(files_id_list, 
                                           sp_destDir, '_context.mfcc')
    
    if splt ==  'train':
        ph_bin = phoneme_binary(list(phone_set))
    
    #lab2binary(labels_file_list, ph_bin, binary_labels_file_list)
    #MVN_normalize(mfcc_file_list, params['mfcc_dim'], norm_mfcc_file_list)
    #make_context_frames(mfcc_file_list, window_mfcc_file_list, n_frames=3, mfcc_dim=39)
    data_desc[splt+'_y'] = binary_labels_file_list
    data_desc[splt+'_x'] = window_mfcc_file_list

  data_desc['ph2bin'] = ph_bin 
  json.dump(data_desc, open(os.path.join(destDir,'dataset.json'),'w'))

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  # IO specs
  parser.add_argument('--src_dir', dest='src_dir', type=str, default='origData', help='src folder within data/ ')
  parser.add_argument('--dest_dir', dest='dest_dir', type=str, default='mvNorm', help='dest folder within data/')
  parser.add_argument('--data_base_dir', dest='data_dir', type=str, default='data', help='data directory base path')
  parser.add_argument('--mfcc_dim', dest='mfcc_dim', type=int, default=39, help='mfcc features dimensions')
  parser.add_argument('--normalization_type', dest='normalization_type', type=str, default='MVN', help='which normalization to use: currently on MVN')

  # dataset_descriptor
  
  # Learning related parmeters

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
