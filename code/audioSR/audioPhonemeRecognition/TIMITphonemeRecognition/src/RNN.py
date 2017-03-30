print('\n\n * Imporing Libaries')
import os
import time;

program_start_time = time.time()

from general_tools import *
from RNN_tools import *

##### SCRIPT META VARIABLES #####
print(' * Setting up ...')

comput_confusion = False
# TODO: ATM this is not implemented


dataRootPath = "/home/matthijs/TCDTIMIT/TIMIT/binary/speech2phonemes26Mels/"
data_path = dataRootPath + "std_preprocess_26_ch.pkl"
# train_data_path = os.path.join(dataRootPath, 'trainData.pkl')
# test_data_path = os.path.join(dataRootPath, 'testData.pkl')

output_path = "/home/matthijs/TCDTIMIT/TIMIT/binary/results"
model_load = os.path.join(output_path, 'best_model.npz')
model_save = os.path.join(output_path, 'best_model')

##### SCRIPT VARIABLES #####
num_epochs = 20

NUM_OUTPUT_UNITS = 61
N_HIDDEN = 275

LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_INIT = 0.1
batch_size = 1

frac_train = 0.7

INPUT_SIZE=26  #num of features to use -> see 'utils.py' in convertToPkl under processDatabase

##### IMPORTIN DATA #####
print('\tdata source: ' + dataRootPath)
print('\tmodel target: ' + model_save + '.npz')

dataset = load_dataset(data_path)
X_train, y_train, X_val, y_val, X_test, y_test = dataset

# X_tr, y_tr, y_tr_onehot= load_dataset(train_data_path,'train')
# X_test, y_test, y_test_onehot = load_dataset(test_data_path,'test')
#
# n_train = int(len(X_tr)*frac_train)
# X_train = X_tr[0:n_train]
# X_val = X_tr[n_train:]
#
# y_train = y_tr[0:n_train]
# y_val = y_tr[n_train:]
#
# dataset = [X_train, y_train, X_val, y_val, X_test, y_test]

##### BUIDING MODEL #####
print(' * Building network ...')
RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size, num_features=INPUT_SIZE, n_hidden=N_HIDDEN,
                            num_output_units=NUM_OUTPUT_UNITS, seed=int(time.time()), debug=False)
RNN_network.load_model(model_load)

##### BUIDING FUNCTION #####
print(" * Compiling functions ...")
RNN_network.build_functions(LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, debug=False)

##### TRAINING #####
print(" * Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                  batch_size=batch_size, compute_confusion=False, debug=False)

print()
print(" * Done")
print()
print('Total time: {:.3f}'.format(time.time() - program_start_time))
