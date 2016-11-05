import numpy as np
import theano
import argparse
import json
import os
from utils.dataprovider import DataProvider
from utils.utils import getModelObj
import re
import io

def main(params):
  # check if having a model_list  
  if params['model_list'] != None:
    with open(params['model_list']) as f:
      model_file_list = f.readlines()
  else:
    model_file_list = [(params['saved_model'])]
  
  # check dp loaded or not to load it once
  dp_loaded = False
  for m in model_file_list:
    m = re.sub("\n","", m)
    cv = json.load(open(m,'r'))
    cv_params = cv['params']
    
    if params['dataset'] != None:
        cv_params['dataset'] = params['dataset']
        cv_params['dataset_desc'] = params['dataset_desc']
    if not dp_loaded:
      dp_loaded = True
      dp = DataProvider(cv_params)
    cv_params['feat_size'] = dp.feat_size
    cv_params['phone_vocab_size'] = dp.phone_vocab
    
    
    # Get the model object and build the model Architecture
    modelObj = getModelObj(cv_params)
    f_train = modelObj.build_model(cv_params)
    modelObj.model.load_weights(cv['weights_file'])
    
    #inpt_x, inpt_y = dp.get_data_array(cv_params['model_type'],[params['split']],cntxt = cv_params['context'])
    #inpt_x, inpt_y = dp.get_data_array(cv_params['model_type'],[params['split']])
    
    splt = params['split']
    if splt == 'eval':
        dataDesc = json.load(open(os.path.join('data', cv_params['dataset'], cv_params['dataDesc']), 'r'))
        ph2bin = dataDesc['ph2bin']
        phoneList = ['']*len(ph2bin)
        for ph in ph2bin:
            phoneList[ph2bin[ph].split().index('1')] = ph

        in_dim = cv_params['in_dim']
        in_file_list = dataDesc[splt+'_x']
        out_file_list = dataDesc[splt+'_y']
        for i in xrange(len(in_file_list)):
            test_x = np.fromfile(in_file_list[i], dtype=np.float32, sep=' ', count=-1)
            test_x.resize(len(test_x)/in_dim, in_dim)
            pred_y = modelObj.model.predict_classes(test_x, batch_size=16)
            out_file = os.path.basename(out_file_list[i])
            with io.open(os.path.join('eval_out', out_file), 'w', encoding='utf-8') as f:
                out_labels = [phoneList[i] for i in pred_y]
                f.write('\n'.join(out_labels))
            print out_file
    #predOut = modelObj.model.predict_classes(inpt_x, batch_size=100)
    predOut = modelObj.model.predict(inpt_x, batch_size=100)
    #accuracy =  100.0*np.sum(predOut == inpt_y.nonzero()[1]) / predOut.shape[0]
    #print('Accuracy of %s the %s set is %0.2f'%(params['saved_model'],params['split'],accuracy))

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  # IO specs
  parser.add_argument('-m','--saved_model', dest='saved_model', type=str, default='', help='input the saved model json file to evluate on')
  parser.add_argument('-s','--split', dest='split', type=str, default='eval', help='which data split to evaluate on')
  
  # Provide these only if evaluating on a dataset other than what the model was trained on
  parser.add_argument('-d','--dataset', dest='dataset', type=str, default=None, help='Which file should we use for read the MFCC features')
  parser.add_argument('--dataset_desc', dest='dataDesc', type=str, default='dataset.json', help='Which file should we use for read the MFCC features')
  
  # models list file
  parser.add_argument('--model_list', dest='model_list', type=str, default=None,\
  help='Text file containing of model files list to evaluate')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
