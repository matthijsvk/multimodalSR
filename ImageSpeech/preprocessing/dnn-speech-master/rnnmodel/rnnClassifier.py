import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, RMSprop
from os import path

def getSolver(params):
    if params['solver'] == 'sgd':
      return SGD(lr=params['lr'], decay=1-params['decay_rate'], momentum=0.9, nesterov=True)
    elif params['solver'] == 'rmsprop':  
      return RMSprop(lr=params['lr'])
      raise ValueError('ERROR in RNN: %s --> This solver type is not yet supported '%(params['solver']))
      
class RnnClassifier:
  def __init__(self, params):
    self.model = Sequential()
    print('----------Using RNN model with the below configuration----------') 
    print('nLayers:%d'%(len(params['hidden_layers'])))
    print('Layer sizes: [%s]'%(' '.join(map(str,params['hidden_layers']))))
    print('Dropout Prob: %.2f '%(params['drop_prob_encoder']))

  def build_model(self, params):
    hidden_layers = params['hidden_layers']
    input_dim = params['feat_size']
    output_dim = params['phone_vocab_size']
    drop_prob = params['drop_prob_encoder']
    self.nLayers = len(hidden_layers)

    # First Layer is an encoder layer
    
    self.model.add(TimeDistributedDense(hidden_layers[0], init='glorot_uniform', input_dim=input_dim))
    self.model.add(Dropout(drop_prob))
    
    # Second Layer is the Recurrent Layer 
    if params.get('recurrent_type','simple') == 'simple':
        self.model.add(SimpleRNN(hidden_layers[1], init='glorot_uniform', inner_init='orthogonal',
            activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=False, 
            input_dim=hidden_layers[0], input_length=None))
    elif params.get('recurrent_type','simple') == 'lstm':
        self.model.add(LSTM(hidden_layers[1], init='glorot_uniform', inner_init='orthogonal',
            input_dim=hidden_layers[0], input_length=None))

    # Then we add dense projection layer to map the RNN outputs to Vocab size 
    self.model.add(Dropout(drop_prob))
    self.model.add(Dense(output_dim, input_dim=hidden_layers[1], init='uniform'))
    self.model.add(Activation('softmax'))
  
    self.solver = getSolver(params)
    self.model.compile(loss='categorical_crossentropy', optimizer=self.solver)
    #score = model.evaluate(test_x)
    self.f_train = self.model.train_on_batch

    return self.f_train

  def train_model(self, train_x, train_y, val_x, val_y,params):
    epoch= params['max_epochs']
    batch_size=params['batch_size']
    out_dir=params['out_dir']
    fname = path.join(out_dir, 'RNN_weights_'+params['out_file_append'] +'_{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    earlystopper= EarlyStopping(monitor='val_loss', patience=params.get('patience',5), verbose=1)
    self.model.fit(train_x, train_y,validation_data=(val_x, val_y), nb_epoch=epoch, batch_size=batch_size, callbacks=[checkpointer, earlystopper])
    return fname, checkpointer.best
      

