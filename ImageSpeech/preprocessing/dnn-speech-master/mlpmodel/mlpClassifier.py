import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from os import path

class MlpClassifier:
  def __init__(self, params):
    self.model = Sequential()
    print('----------Using MLP model with the below configuration----------') 
    print('nLayers:%d'%(len(params['hidden_layers'])))
    print('Layer sizes: [%s]'%(' '.join(map(str,params['hidden_layers']))))
    print('Dropout Prob: %.2f '%(params['drop_prob_encoder']))

  def build_model(self, params):
    hidden_layers = params['hidden_layers']
    input_dim = params['feat_size']
    output_dim = params['phone_vocab_size']
    drop_prob = params['drop_prob_encoder']
    self.nLayers = len(hidden_layers)
    # first layer takes input data
    self.model.add(Dense(hidden_layers[0], input_dim=input_dim, init='uniform'))
    self.model.add(Activation('sigmoid'))
    self.model.add(Dropout(drop_prob))
    # hidden layers
    for i in xrange(1,len(hidden_layers)):
        self.model.add(Dense(hidden_layers[i], input_dim=hidden_layers[i-1],
            init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drop_prob))

    #output layer
    self.model.add(Dense(output_dim, input_dim=hidden_layers[-1], init='uniform'))
    self.model.add(Activation('softmax'))
  
    if params['solver'] == 'sgd':
      self.solver = SGD(lr=params['lr'], decay=1-params['decay_rate'], momentum=0.9, nesterov=True)
    else:  
      raise ValueError('ERROR in MLP: %s --> This solver type is not yet supported '%(params['solver']))
      
    self.model.compile(loss='categorical_crossentropy', optimizer=self.solver)
    #score = model.evaluate(test_x)
    self.f_train = self.model.train_on_batch

    return self.f_train

  def train_model(self, train_x, train_y, val_x, val_y,params):
    epoch= params['max_epochs']
    batch_size=params['batch_size']
    out_dir=params['out_dir']
    fname = path.join(out_dir, 'MLP_weights_'+params['out_file_append'] +'_{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    self.model.fit(train_x, train_y,validation_data=(val_x, val_y), nb_epoch=epoch, batch_size=batch_size, callbacks=[checkpointer])
    return fname, checkpointer.best
      

